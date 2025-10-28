import os
import sys
import time
import socket
import subprocess
import threading
import webbrowser

HOST = "127.0.0.1"
PORT = 8080
INDEX_URL = f"http://{HOST}:{PORT}/"

APP_DIR = os.path.dirname(os.path.abspath(__file__))
SPLASH_FILE = os.path.join(APP_DIR, "splash.html")

def is_frozen():
    return getattr(sys, "frozen", False)

def is_port_open(host, port):
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False

def start_server():
    """
    Launch uvicorn as a child process, same as before.
    We DO NOT loop-open browsers here. We just start the API.
    """
    python_exec = sys.executable  # venv python in dev, bundled exe when frozen
    cmd = [
        python_exec,
        "-m", "uvicorn",
        "app:app",
        "--host", HOST,
        "--port", str(PORT),
    ]
    print(f"[launcher] starting server: {cmd} (cwd={APP_DIR})")
    proc = subprocess.Popen(
        cmd,
        cwd=APP_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc

def build_splash_url(phase="initializing", pct=0, err=""):
    import urllib.parse
    q = {"phase": phase, "pct": str(pct)}
    if err:
        q["err"] = err
    return "file://" + SPLASH_FILE + "?" + urllib.parse.urlencode(q)

def show_splash_window(initial_phase="initializing", initial_pct=0, err=""):
    """
    Show ONE webview window with splash.html.
    In frozen builds macOS can get cranky, so we can skip splash when frozen
    to stay safe. Dev only for now.
    """
    if is_frozen():
        print("[launcher] frozen build: skipping splash window for safety")
        return None

    try:
        import webview  # pywebview
    except ImportError:
        print("[launcher] pywebview not available, skipping splash")
        return None

    start_url = build_splash_url(initial_phase, initial_pct, err)
    win = webview.create_window(
        "Calypso Labs — Boot",
        start_url,
        width=900,
        height=500,
        resizable=False,
        fullscreen=False,
        confirm_close=False,
    )

    def _run_webview():
        try:
            webview.start()  # blocking call, so run it in a thread
        except Exception as e:
            print(f"[launcher] splash crashed: {e}")

    t = threading.Thread(target=_run_webview, daemon=True)
    t.start()
    return win

def main():
    print("[launcher] launch requested")
    print(f"[launcher] running from {APP_DIR}")
    print(f"[launcher] python_exec is {sys.executable}")
    print(f"[launcher] frozen? {is_frozen()}")

    splash_win = show_splash_window(initial_phase="initializing", initial_pct=5)

    proc = start_server()
    print(f"[launcher] server pid={proc.pid}")

    pct = 5
    phase = "initializing"

    # loop: wait for port OR crash
    while True:
        # did server die?
        if proc.poll() is not None:
            print("[launcher] server exited before ready")
            # dump logs for debug
            out, err = proc.communicate()
            print("----- server logs begin -----")
            print(out)
            print(err)
            print("----- server logs end -----")
            return  # do not open browser

        # is it up?
        if is_port_open(HOST, PORT):
            print("[launcher] server is up!")
            phase = "ready"
            pct = 100
            break

        # not up yet, bump pct but clamp to 95
        pct = min(pct + 5, 95)
        if pct < 50:
            phase = "initializing"
        else:
            phase = "starting_server"

        time.sleep(0.5)

    # at this point server is listening. open ONE browser tab.
    print(f"[launcher] opening browser {INDEX_URL}")
    webbrowser.open_new_tab(INDEX_URL)

    # NOTE: we are NOT spawning another launcher, we are not re-calling ourselves,
    # and we are not reopening the browser in a loop. Just once.

    # keep process alive so server stays up
    try:
        while proc.poll() is None:
            time.sleep(1.0)
    finally:
        print("[launcher] shutting down server")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

if __name__ == "__main__":
    main()
