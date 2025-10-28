import os
import sys
import time
import socket
import subprocess
import webbrowser

import webview  # pywebview


HOST = "127.0.0.1"
PORT = 8080
INDEX_URL = f"http://{HOST}:{PORT}/"
APP_DIR = os.path.dirname(os.path.abspath(__file__))


def is_port_open(host, port):
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def start_server():
    """
    Launch uvicorn subprocess and return Popen handle.
    """
    python_exec = sys.executable  # should be venv/bin/python3 if venv is active
    cmd = [
        python_exec,
        "-m", "uvicorn",
        "app:app",
        "--host", HOST,
        "--port", str(PORT),
    ]

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    proc = subprocess.Popen(
        cmd,
        cwd=APP_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc


def wait_for_server(proc, timeout_sec=20):
    """
    Wait for the server socket to listen.
    Return True if up, False if crash/timeout.
    """
    start_t = time.time()
    while True:
        # Did server die?
        if proc.poll() is not None:
            print("[launcher] server exited early")
            return False

        # Is port open?
        if is_port_open(HOST, PORT):
            print("[launcher] server is UP")
            return True

        # Timeout?
        if time.time() - start_t > timeout_sec:
            print("[launcher] timeout waiting for server")
            return False

        time.sleep(0.4)


def build_splash_url(phase="initializing", pct=5, err=""):
    import urllib.parse
    splash_path = os.path.join(APP_DIR, "splash.html")
    q = {"phase": phase, "pct": str(pct)}
    if err:
        q["err"] = err
    return "file://" + splash_path + "?" + urllib.parse.urlencode(q)


def show_splash_and_close_after(delay_sec, phase, pct, err=""):
    """
    Create splash window, run webview.start() with a callback that will
    sleep a moment, then destroy this specific window.
    This call BLOCKS until the window is destroyed.
    """
    splash_url = build_splash_url(phase=phase, pct=pct, err=err)

    win = webview.create_window(
        "Calypso Labs — Boot",
        splash_url,
        width=900,
        height=500,
        resizable=False,
        fullscreen=False,
        confirm_close=False,
    )

    def close_after(win_obj):
        # win_obj is the same window we created.
        time.sleep(delay_sec)
        try:
            win_obj.destroy()
        except Exception as e:
            print(f"[launcher] splash destroy error (harmless): {e}")

    # This will block until win_obj.destroy() is called.
    webview.start(close_after, win)


def babysit_server(proc):
    """
    Keep launcher alive as long as uvicorn is running.
    When launcher ends (Ctrl+C), kill uvicorn.
    """
    print("[launcher] babysitting server process")
    try:
        while proc.poll() is None:
            time.sleep(1.0)
    finally:
        print("[launcher] shutting down server process")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("[launcher] exiting cleanly")


def main():
    print("[launcher] launch requested")

    # 1. start backend
    srv = start_server()
    print(f"[launcher] server pid={srv.pid}")

    # 2. wait for ready
    up = wait_for_server(srv, timeout_sec=20)

    if up:
        # 3a. happy path splash: show "Complete 100%" ~0.8s then auto-close
        show_splash_and_close_after(
            delay_sec=0.8,
            phase="complete",
            pct=100,
            err=""
        )

        # 4. open browser tab
        print(f"[launcher] opening browser {INDEX_URL}")
        webbrowser.open_new_tab(INDEX_URL)

        # 5. babysit until user stops launcher
        babysit_server(srv)

    else:
        # 3b. error splash: show failure for ~3s, then close and quit
        show_splash_and_close_after(
            delay_sec=3.0,
            phase="error",
            pct=5,
            err="Server failed to start. Please quit and relaunch."
        )
        # don't babysit, server is dead anyway


if __name__ == "__main__":
    main()
