import os
import sys
import time
import socket
import threading
import subprocess

import webview

HOST = "127.0.0.1"
PORT = 8080

# -----------------------
# helpers
# -----------------------

def is_port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.3):
            return True
    except OSError:
        return False

def is_frozen() -> bool:
    return getattr(sys, "frozen", False)

def wait_for_server(timeout_sec=15):
    start = time.time()
    pct = 0
    while True:
        if is_port_open(HOST, PORT):
            print("[launcher] server is UP")
            return True
        if time.time() - start > timeout_sec:
            print("[launcher] ERROR: timeout waiting for server")
            return False
        pct += 5
        print(f"[launcher] waiting for server... {pct}%")
        time.sleep(0.3)

def pump_logs(proc):
    """Mirror the uvicorn subprocess output (dev mode only)."""
    def _pump(stream, label):
        for line in iter(stream.readline, ''):
            if not line:
                break
            print(f"[server:{label}] {line.rstrip()}")
    threading.Thread(target=_pump, args=(proc.stdout, "stdout"), daemon=True).start()
    threading.Thread(target=_pump, args=(proc.stderr, "stderr"), daemon=True).start()

# -----------------------
# DEV MODE server launcher (subprocess)
# -----------------------

def start_server_dev():
    """Run uvicorn as a subprocess using the venv python (dev)."""
    app_dir = os.path.dirname(os.path.abspath(__file__))
    python_exec = sys.executable  # this will be your venv/bin/python3

    cmd = [
        python_exec,
        "-m", "uvicorn",
        "app:app",
        "--host", HOST,
        "--port", str(PORT),
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

# -----------------------
# FROZEN MODE server launcher (in-process thread)
# -----------------------

def _run_uvicorn_inprocess():
    """Target for background thread in frozen mode."""
    # we import here so PyInstaller can collect uvicorn & app deps
    import uvicorn
    # IMPORTANT: app.py must be bundled and importable
    # PyInstaller normally adds the bundle dir to sys.path
    uvicorn.run(
        "app:app",
        host=HOST,
        port=PORT,
        log_level="info",
    )

def start_server_frozen_thread():
    """Start uvicorn in a thread instead of spawning a new exe."""
    t = threading.Thread(target=_run_uvicorn_inprocess, daemon=True)
    t.start()
    return t  # we won't kill it manually; closing the app exits process anyway

# -----------------------
# main UI flow
# -----------------------

def main():
    print("[launcher] launch requested")
    mode = "frozen" if is_frozen() else "dev"
    print(f"[launcher] mode = {mode}")

    server_proc = None
    server_thread = None

    if mode == "dev":
        # spawn uvicorn in a subprocess (your current happy path)
        server_proc = start_server_dev()
        pump_logs(server_proc)
    else:
        # run uvicorn in-process thread
        server_thread = start_server_frozen_thread()

    # wait for port 8080 to be live
    if not wait_for_server(timeout_sec=20):
        print("[launcher] backend failed, terminating")
        if server_proc:
            try:
                server_proc.terminate()
            except Exception:
                pass
        return

    app_url = f"http://{HOST}:{PORT}/"
    print(f"[launcher] showing client window @ {app_url}")

    window = webview.create_window(
        title="Calypso Labs — AI Text Scanner",
        url=app_url,
        width=1100,
        height=750,
        resizable=True,
        confirm_close=True,
    )

    # cocoa is correct on macOS
    webview.start(gui="cocoa")

    # when the window closes:
    print("[launcher] window closed, shutting down server...")

    if server_proc:
        # dev: stop subprocess cleanly
        try:
            server_proc.terminate()
            server_proc.wait(timeout=5)
        except Exception:
            try:
                server_proc.kill()
            except Exception:
                pass

    print("[launcher] clean exit")

if __name__ == "__main__":
    main()
