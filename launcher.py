import os
import sys
import time
import socket
import threading
import subprocess
import webview

HOST = "127.0.0.1"

# -----------------------
# helpers
# -----------------------

def pick_free_port(host: str) -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, 0))  # 0 = let OS choose
    port = s.getsockname()[1]
    s.close()
    return port

def is_port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.3):
            return True
    except OSError:
        return False

def is_frozen() -> bool:
    return getattr(sys, "frozen", False)

def wait_for_server(host: str, port: int, timeout_sec: int = 60) -> bool:
    start = time.time()
    while True:
        if is_port_open(host, port):
            print("[launcher] server is UP")
            return True
        if time.time() - start > timeout_sec:
            print("[launcher] ERROR: timeout waiting for server")
            return False
        time.sleep(0.3)

def pump_logs(proc):
    """Mirror uvicorn subprocess output in dev mode."""
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

def start_server_dev(port: int):
    """
    Run uvicorn as a subprocess using the current interpreter (venv python).
    This is only used when we're NOT frozen.
    """
    app_dir = os.path.dirname(os.path.abspath(__file__))
    python_exec = sys.executable  # e.g. .../venv/bin/python3

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

# -----------------------
# FROZEN MODE server launcher (in-process thread)
# -----------------------

def _run_uvicorn_inprocess(port: int):
    """
    Run uvicorn in this same process/thread pool.
    This avoids trying to exec the bundled binary as if it were `python -m`.
    """
    import uvicorn
    uvicorn.run(
        "app:app",
        host=HOST,
        port=port,
        log_level="info",
    )

def start_server_frozen_thread(port: int):
    """
    Launch uvicorn in a daemon thread when we're in a frozen app bundle.
    """
    t = threading.Thread(target=_run_uvicorn_inprocess, args=(port,), daemon=True)
    t.start()
    return t

# -----------------------
# main UI flow
# -----------------------

def main():
    print("[launcher] launch requested")
    mode = "frozen" if is_frozen() else "dev"
    print(f"[launcher] mode = {mode}")

    port = pick_free_port(HOST)
    print(f"[launcher] selected port = {port}")

    server_proc = None

    if mode == "dev":
        server_proc = start_server_dev(port)
        pump_logs(server_proc)
    else:
        start_server_frozen_thread(port)

    # wait until 127.0.0.1:<port> is live (or give up)
    if not wait_for_server(HOST, port, timeout_sec=60):
        print("[launcher] backend failed, terminating")
        if server_proc:
            try:
                server_proc.terminate()
            except Exception:
                pass
        return

    app_url = f"http://{HOST}:{port}/"
    print(f"[launcher] showing client window @ {app_url}")

    webview.create_window(
        title="Calypso Labs — AI Text Scanner",
        url=app_url,
        width=1100,
        height=750,
        resizable=True,
        confirm_close=True,
    )

    # cocoa for macOS
    webview.start(gui="cocoa")

    # when GUI window closes:
    print("[launcher] window closed, shutting down server...")

    if server_proc:
        # dev mode cleanup
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
