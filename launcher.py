# launcher.py — CopyCat launcher (native window only; no browser fallback)

import os
import sys
import time
import socket
import threading
import subprocess
import pathlib
import platform

HOST = "127.0.0.1"

# -----------------------
# bundle path helpers
# -----------------------

def _set_cwd_to_bundle_root() -> None:
    """
    When frozen, sys._MEIPASS is the extracted bundle resource dir.
    In dev, use the directory containing this file.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        os.chdir(sys._MEIPASS)
    else:
        os.chdir(pathlib.Path(__file__).resolve().parent)


def _ensure_bundle_on_syspath() -> None:
    """
    In frozen mode, app.py and resources are unpacked to sys._MEIPASS.
    Ensure it's on sys.path so 'import app' works.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        meipass = sys._MEIPASS
        if meipass not in sys.path:
            sys.path.insert(0, meipass)


_set_cwd_to_bundle_root()
_ensure_bundle_on_syspath()

# -----------------------
# networking helpers
# -----------------------

def pick_free_port(host: str) -> int:
    """
    Bind to port 0 to let the OS choose a free ephemeral port.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, 0))
    port = s.getsockname()[1]
    s.close()
    return port


def is_http_responding(host: str, port: int) -> bool:
    """
    Confirm the backend is not only listening, but actually responding to HTTP.
    (This avoids the 'port open but request hangs' situation.)
    """
    try:
        with socket.create_connection((host, port), timeout=0.5) as sock:
            req = f"GET / HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n"
            sock.sendall(req.encode("ascii"))
            sock.settimeout(0.8)
            data = sock.recv(16)
            return bool(data)
    except OSError:
        return False


def wait_for_server(host: str, port: int, timeout_sec: int = 60) -> bool:
    start = time.time()
    while True:
        if is_http_responding(host, port):
            print("[launcher] server is UP (HTTP responding)")
            return True
        if time.time() - start > timeout_sec:
            print("[launcher] ERROR: timeout waiting for server")
            return False
        time.sleep(0.25)


def is_frozen() -> bool:
    return getattr(sys, "frozen", False)

# -----------------------
# backend launchers
# -----------------------

def _run_backend_blocking(host: str, port: int) -> int:
    """
    Run uvicorn in-process (used for --backend mode).
    """
    import uvicorn
    import app as app_module

    config = uvicorn.Config(
        app_module.app,
        host=host,
        port=port,
        log_level="info",
        access_log=False,
    )
    uvicorn.Server(config).run()
    return 0


def start_server_dev(port: int) -> subprocess.Popen:
    """
    Dev mode: run uvicorn as a subprocess using the active interpreter.
    """
    app_dir = os.path.dirname(os.path.abspath(__file__))
    python_exec = sys.executable

    cmd = [
        python_exec,
        "-m", "uvicorn",
        "app:app",
        "--host", HOST,
        "--port", str(port),
    ]

    print(f"[launcher] starting server (dev): {cmd} (cwd={app_dir})")
    proc = subprocess.Popen(
        cmd,
        cwd=app_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc


def pump_logs(proc: subprocess.Popen) -> None:
    """
    Mirror backend subprocess output in dev mode.
    """
    def _pump(stream, label):
        try:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                print(f"[server:{label}] {line.rstrip()}")
        except Exception:
            pass

    if proc.stdout:
        threading.Thread(target=_pump, args=(proc.stdout, "stdout"), daemon=True).start()
    if proc.stderr:
        threading.Thread(target=_pump, args=(proc.stderr, "stderr"), daemon=True).start()

# -----------------------
# main
# -----------------------

def main() -> int:
    # Backend worker mode (spawned by frozen launcher)
    if "--backend" in sys.argv:
        try:
            i = sys.argv.index("--port")
            port = int(sys.argv[i + 1])
        except Exception:
            print("[backend] FATAL: missing --port")
            return 2

        print(f"[backend] starting on {HOST}:{port}")
        return _run_backend_blocking(HOST, port)

    print("[launcher] launch requested")
    print(f"[launcher] platform={sys.platform} machine={platform.machine()} frozen={is_frozen()}")

    mode = "frozen" if is_frozen() else "dev"
    print(f"[launcher] mode = {mode}")

    port = pick_free_port(HOST)
    print(f"[launcher] selected port = {port}")

    server_proc = None

    if mode == "dev":
        server_proc = start_server_dev(port)
        pump_logs(server_proc)
    else:
        # Frozen: run backend in a subprocess to avoid UI/GIL deadlocks.
        cmd = [sys.executable, "--backend", "--port", str(port)]
        print(f"[launcher] starting server (frozen subprocess): {cmd}")
        server_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    if not wait_for_server(HOST, port, timeout_sec=60):
        print("[launcher] backend failed to become ready; terminating")
        if server_proc:
            try:
                server_proc.terminate()
            except Exception:
                pass
        return 1

    app_url = f"http://{HOST}:{port}/"
    print(f"[launcher] showing native window @ {app_url}")

    # NATIVE WINDOW ONLY (same behavior as arm64).
    # If this fails on x86_64, we WANT a traceback so we can fix packaging/signing.
    try:
        import webview

        webview.create_window(
            title="Calypso Labs — AI Text Scanner",
            url=app_url,
            width=1100,
            height=750,
            resizable=True,
            confirm_close=True,
        )
        webview.start(gui="cocoa")
    except Exception:
        import traceback
        print("[launcher] FATAL: failed to start native Cocoa webview UI.")
        traceback.print_exc()
        return 2
    finally:
        print("[launcher] window closed, shutting down server...")

        if server_proc:
            try:
                server_proc.terminate()
                server_proc.wait(timeout=5)
            except Exception:
                try:
                    server_proc.kill()
                except Exception:
                    pass

    print("[launcher] clean exit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
