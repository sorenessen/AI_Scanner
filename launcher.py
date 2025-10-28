import os
import sys
import time
import socket
import subprocess
import threading
import webview  # pywebview

HOST = "127.0.0.1"
PORT = 8080

def is_port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.3):
            return True
    except OSError:
        return False

def start_server():
    """
    Launch uvicorn server as a subprocess.
    We do NOT use reload, we do NOT daemonize.
    We capture the process so we can kill it when the window closes.
    """
    python_exec = sys.executable  # this will be venv/bin/python3 when run in venv
    cmd = [
        python_exec,
        "-m", "uvicorn",
        "app:app",
        "--host", HOST,
        "--port", str(PORT),
    ]

    print(f"[launcher] starting server: {cmd}")
    proc = subprocess.Popen(
        cmd,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc

def wait_for_server(timeout_sec=10):
    """
    Poll for the server to start listening on PORT.
    timeout_sec = how long we wait before giving up.
    Returns True if the port opened, False otherwise.
    """
    start_t = time.time()
    pct = 0
    while True:
        if is_port_open(HOST, PORT):
            print("[launcher] server is UP")
            return True

        if time.time() - start_t > timeout_sec:
            print("[launcher] ERROR: server did not come up in time")
            return False

        pct += 5
        print(f"[launcher] waiting for server... {pct}%")
        time.sleep(0.3)

def pump_server_logs(proc):
    """
    Background thread: read server stdout/stderr and mirror to console
    so you can debug without the window hiding crashes.
    """
    def _pump(stream, label):
        for line in iter(stream.readline, ''):
            if not line:
                break
            print(f"[server:{label}] {line.rstrip()}")
    t_out = threading.Thread(target=_pump, args=(proc.stdout, "stdout"), daemon=True)
    t_err = threading.Thread(target=_pump, args=(proc.stderr, "stderr"), daemon=True)
    t_out.start()
    t_err.start()

def main():
    print("[launcher] launch requested")

    # 1. start backend server
    proc = start_server()
    pump_server_logs(proc)

    # 2. wait until it's actually listening
    if not wait_for_server(timeout_sec=10):
        print("[launcher] FATAL: backend never started, killing process and exiting")
        try:
            proc.terminate()
        except Exception:
            pass
        return

    # 3. create ONE desktop window that points at the local app
    app_url = f"http://{HOST}:{PORT}/"
    print(f"[launcher] creating client window at {app_url}")

    window = webview.create_window(
        title="Calypso Labs — AI Text Scanner",
        url=app_url,
        width=1000,
        height=700,
        resizable=True,
        fullscreen=False,
        confirm_close=True,  # we'll intercept close to stop the server
    )

    # This runs on the main thread and blocks until the window closes.
    # When it returns, user closed the window -> we stop the server.
    def on_closed():
        print("[launcher] window closed, shutting down server...")
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        print("[launcher] goodbye.")

    # pywebview doesn't have a "closed" callback param in create_window,
    # but we can hook into gui exit by running webview.start() with a callback.
    #
    # The "func" we pass to start() will run *after* the window is ready,
    # but start() itself doesn't return until the window is closed.
    # So we wrap start() and then call on_closed() after start() returns.
    #
    def run_gui():
        # start the GUI loop (blocks until user closes the window)
        webview.start(gui='cocoa')  # cocoa = macOS native

    try:
        run_gui()
    finally:
        on_closed()

if __name__ == "__main__":
    main()
