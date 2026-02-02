#!/usr/bin/env python3
"""
run_copycat_desktop.py

Desktop launcher for CopyCat:
- Starts the FastAPI/uvicorn backend on a free localhost port
- Waits for /healthz
- Opens the UI in a pywebview window
- Writes two logs:
  1) ~/Library/Logs/CopyCat/launch.log   (bootstrap + uncaught exceptions)
  2) ~/Library/Logs/CopyCat/copycat.log  (runtime/server/health status)

This is a full drop-in replacement.
"""

# -------------------------
# CopyCat bootstrap log (always on)
# -------------------------
import os
import sys
import time
import socket
import threading
import traceback
import datetime
import pathlib
from pathlib import Path

LOG_DIR = Path.home() / "Library" / "Logs" / "CopyCat"
LOG_DIR.mkdir(parents=True, exist_ok=True)

BOOT_LOG = LOG_DIR / "launch.log"
RUNTIME_LOG = LOG_DIR / "copycat.log"


def _boot_log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(BOOT_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")


def _runtime_log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with RUNTIME_LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")


def _reveal_logs_in_finder() -> None:
    try:
        os.system(f'open -R "{RUNTIME_LOG}"')
    except Exception:
        pass


_boot_log("=== CopyCat starting ===")
_boot_log(f"argv={sys.argv}")
_boot_log(f"cwd={os.getcwd()}")
_boot_log(f"executable={sys.executable}")
_boot_log(f"python={sys.version.replace(os.linesep, ' ')}")
_boot_log(f"frozen={getattr(sys, 'frozen', False)}")
_boot_log(f"MEIPASS={getattr(sys, '_MEIPASS', None)}")

try:
    import platform

    _boot_log(f"platform={platform.platform()}")
    _boot_log(f"machine={platform.machine()}")
except Exception:
    pass


def _excepthook(exctype, value, tb):
    _boot_log("UNCAUGHT EXCEPTION:")
    _boot_log("".join(traceback.format_exception(exctype, value, tb)))
    _reveal_logs_in_finder()
    sys.__excepthook__(exctype, value, tb)


sys.excepthook = _excepthook

# -------------------------
# Bundle path handling
# -------------------------
HERE = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)).resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

_boot_log(f"HERE={HERE}")

# -------------------------
# Imports that depend on sys.path adjustments
# -------------------------
try:
    import webview
    import uvicorn
    import app  # ensures PyInstaller includes app.py
except Exception:
    _boot_log("IMPORT ERROR:")
    _boot_log(traceback.format_exc())
    _reveal_logs_in_finder()
    raise


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _probe_expected_files() -> None:
    """
    Log what we can see from the bundle runtime.
    This helps catch the common 'bounces then quits' case where data files
    did not get bundled where the app expects them.
    """
    candidates = [
        ("index.html (cwd)", Path("index.html")),
        ("static (cwd)", Path("static")),
        ("model_centroids (cwd)", Path("model_centroids")),
        ("pd_sources (cwd)", Path("pd_sources")),
        ("pd_fingerprints (cwd)", Path("pd_fingerprints")),
        ("index.html (HERE)", HERE / "index.html"),
        ("static (HERE)", HERE / "static"),
        ("model_centroids (HERE)", HERE / "model_centroids"),
        ("pd_sources (HERE)", HERE / "pd_sources"),
        ("pd_fingerprints (HERE)", HERE / "pd_fingerprints"),
    ]

    _runtime_log("---- bundle/file probes ----")
    for label, p in candidates:
        try:
            _runtime_log(f"{label}: exists={p.exists()} path={p}")
        except Exception as e:
            _runtime_log(f"{label}: probe failed: {repr(e)}")


def start_server(port: int) -> None:
    """
    Runs uvicorn in the background thread.
    """
    try:
        _runtime_log(f"Starting uvicorn on 127.0.0.1:{port}")

        # app.app should be your FastAPI instance; keep this consistent with your codebase.
        uvicorn.run(
            app.app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            access_log=False,
        )

        _runtime_log("Uvicorn exited normally (unexpected for desktop app).")
    except Exception:
        _runtime_log("Uvicorn crashed:")
        _runtime_log(traceback.format_exc())
        _reveal_logs_in_finder()
        raise


def wait_for_health(url: str, timeout_s: float = 240.0) -> bool:
    """
    Avoid external deps (requests). Use urllib.
    """
    import urllib.request

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                # healthz should return 200-ish if app is ready
                if 200 <= resp.status < 300:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def show_error_window(health_url: str) -> None:
    """
    Display a simple HTML page showing where logs are.
    """
    html = f"""
    <html>
      <body style="font-family: -apple-system, system-ui; padding: 20px; line-height: 1.35;">
        <h2>CopyCat failed to start</h2>
        <p>The local server never became ready at:</p>
        <pre style="background:#f6f6f6; padding:10px; border-radius:8px;">{health_url}</pre>

        <p>Log files:</p>
        <pre style="background:#f6f6f6; padding:10px; border-radius:8px;">
{BOOT_LOG}
{RUNTIME_LOG}
        </pre>

        <p>Please send those logs to diagnose the issue.</p>
      </body>
    </html>
    """
    webview.create_window("CopyCat", html=html, width=900, height=520, resizable=True)
    webview.start()


if __name__ == "__main__":
    try:
        _runtime_log("==== CopyCat launch ====")
        _runtime_log(f"HERE={HERE}")
        _runtime_log(f"cwd={os.getcwd()}")
        _runtime_log(f"Python={sys.version.replace(os.linesep, ' ')}")
        _runtime_log(f"frozen={getattr(sys, 'frozen', False)}")
        _runtime_log(f"MEIPASS={getattr(sys, '_MEIPASS', None)}")

        _probe_expected_files()

        port = pick_free_port()
        base_url = f"http://127.0.0.1:{port}"
        health_url = f"{base_url}/healthz"
        _runtime_log(f"Port={port}")
        _runtime_log(f"Health URL={health_url}")

        # Start server in background
        t = threading.Thread(target=start_server, args=(port,), daemon=True)
        t.start()

        ok = wait_for_health(health_url, timeout_s=240.0)
        if not ok:
            _runtime_log("Health check never became ready. Showing error window.")
            _reveal_logs_in_finder()
            show_error_window(health_url)
            sys.exit(1)

        _runtime_log("Health check OK. Opening webview.")
        webview.create_window(
            "CopyCat",
            base_url,
            width=1100,
            height=760,
            resizable=True,
        )
        webview.start()

    except Exception:
        _runtime_log("Launcher crashed:")
        _runtime_log(traceback.format_exc())
        _reveal_logs_in_finder()
        raise
