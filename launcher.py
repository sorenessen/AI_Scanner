import os
import socket
import subprocess
import sys
import time

HOST = "127.0.0.1"
PORT = 8080

# --- Where is app.py? ---
# We assume launcher.py lives in the same folder as app.py (project root).
APP_DIR = os.path.dirname(os.path.abspath(__file__))

def is_frozen():
    # True when we're running from a PyInstaller bundle
    return getattr(sys, "frozen", False)

def is_port_open(host, port):
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False

def start_server():
    """
    Start uvicorn app:app on HOST:PORT.
    In dev: use this same python (venv python).
    In frozen: sys.executable will be the bundled binary.
    """

    python_exec = sys.executable  # this is venv/bin/python3.12 in dev, or the bundle exe when frozen

    # Clean env so PyInstaller shims don't get confused later.
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    cmd = [
        python_exec,
        "-m", "uvicorn",
        "app:app",
        "--host", HOST,
        "--port", str(PORT),
    ]

    print(f"[launcher] starting server with cmd={cmd} cwd={APP_DIR}", flush=True)

    proc = subprocess.Popen(
        cmd,
        cwd=APP_DIR,            # <- important so `import app` etc. works
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    return proc

def wait_for_server(proc, timeout_seconds=20):
    """
    Wait until:
    - port is open (success), OR
    - proc dies (fail), OR
    - timeout
    Returns True if server is up, False otherwise.
    """

    start_time = time.time()
    pct = 0

    while True:
        # if crashed, dump output and bail
        if proc.poll() is not None:
            print("[launcher] server exited early, dumping logs:", flush=True)
            # read whatever output remains
            leftover = proc.stdout.read()
            if leftover:
                print("----- server logs begin -----", flush=True)
                print(leftover, flush=True)
                print("----- server logs end -----", flush=True)
            return False

        # if listening, success
        if is_port_open(HOST, PORT):
            print("[launcher] server is listening", flush=True)
            return True

        # timeout?
        if time.time() - start_time > timeout_seconds:
            print("[launcher] timeout waiting for server to listen, dumping logs:", flush=True)
            leftover = proc.stdout.read()
            if leftover:
                print("----- server logs begin -----", flush=True)
                print(leftover, flush=True)
                print("----- server logs end -----", flush=True)
            return False

        pct += 5
        print(f"[launcher] waiting for server... {pct}%", flush=True)
        time.sleep(0.5)

def pump_logs_until_exit(proc):
    """
    Keep reading server output line by line so you can see runtime logs.
    Ctrl+C will stop both launcher and server.
    """
    try:
        while proc.poll() is None:
            line = proc.stdout.readline()
            if line:
                print("[server]", line.rstrip(), flush=True)
            else:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("[launcher] Ctrl+C received, shutting down server", flush=True)
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

def main():
    print("[launcher] launch requested", flush=True)
    print(f"[launcher] running from {APP_DIR}", flush=True)
    print(f"[launcher] python_exec is {sys.executable}", flush=True)
    print(f"[launcher] frozen? {is_frozen()}", flush=True)

    proc = start_server()
    print(f"[launcher] server proc pid={proc.pid}", flush=True)

    up = wait_for_server(proc)
    if not up:
        print("[launcher] cannot continue", flush=True)
        return

    # success
    print("")
    print("========================================", flush=True)
    print(" AI Text Scanner is now running at:", flush=True)
    print(f"   http://{HOST}:{PORT}/", flush=True)
    print("")
    print(" Open that URL in your browser.", flush=True)
    print(" Leave this terminal open while you use the app.", flush=True)
    print(" Press Ctrl+C here to quit the app.", flush=True)
    print("========================================", flush=True)
    print("")

    pump_logs_until_exit(proc)

if __name__ == "__main__":
    main()
