import os
import sys
import time
import threading
import socket
from pathlib import Path

# Ensure local imports work when bundled
HERE = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)).resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import webview
import uvicorn
import app  # forces PyInstaller to include app.py


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def log_path() -> Path:
    # ~/Library/Logs/CopyCat/copycat.log on macOS
    base = Path.home() / "Library" / "Logs" / "CopyCat"
    base.mkdir(parents=True, exist_ok=True)
    return base / "copycat.log"


def write_log(msg: str) -> None:
    p = log_path()
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with p.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")


def start_server(port: int):
    try:
        write_log(f"Starting uvicorn on 127.0.0.1:{port}")
        uvicorn.run(app.app, host="127.0.0.1", port=port, log_level="warning")
        write_log("Uvicorn exited normally (unexpected for desktop app).")
    except Exception as e:
        write_log(f"Uvicorn crashed: {repr(e)}")
        raise


def wait_for_health(url: str, timeout_s: float = 180.0) -> bool:
    # Avoid extra deps (requests). Use urllib.
    import urllib.request

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if 200 <= resp.status < 300:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


if __name__ == "__main__":
    try:
        # Choose a free port so we never collide with someone’s existing 8000
        port = pick_free_port()
        base_url = f"http://127.0.0.1:{port}"
        health_url = f"{base_url}/healthz"

        write_log("==== CopyCat launch ====")
        write_log(f"HERE={HERE}")
        write_log(f"Python={sys.version}")
        write_log(f"Port={port}")

        # Start server in background
        t = threading.Thread(target=start_server, args=(port,), daemon=True)
        t.start()

        # Wait for server readiness (Intel can be slow to load model)
        ok = wait_for_health(health_url, timeout_s=240.0)
        if not ok:
            write_log("Health check never became ready. Showing error page.")
            # Show a simple HTML error with the log path for debugging
            html = f"""
            <html><body style="font-family: -apple-system, system-ui; padding: 20px;">
            <h2>CopyCat failed to start</h2>
            <p>The local server never became ready at <code>{health_url}</code>.</p>
            <p>Log file:</p>
            <pre>{log_path()}</pre>
            <p>If you send that log file, we can pinpoint the crash.</p>
            </body></html>
            """
            webview.create_window("CopyCat", html=html, width=900, height=520, resizable=True)
            webview.start()
            sys.exit(1)

        # Open UI once backend is actually ready
        write_log("Health check OK. Opening webview.")
        webview.create_window(
            "CopyCat",
            base_url,
            width=1100,
            height=760,
            resizable=True,
        )
        webview.start()

    except Exception as e:
        write_log(f"Launcher crashed: {repr(e)}")
        raise
