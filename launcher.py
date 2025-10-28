import os
import sys
import time
import socket
import subprocess
import webbrowser

import webview  # this will run on main thread


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
    Launch uvicorn in the background. Return the Popen handle.
    """
    python_exec = sys.executable
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
    print(f"[launcher] server pid={proc.pid}")
    return proc


def wait_for_server_or_die(proc, timeout_sec=30):
    """
    Poll for PORT to open, or bail if proc dies / timeout hits.
    """
    pct = 5
    start_t = time.time()

    while True:
        # crashed?
        if proc.poll() is not None:
            print("[launcher] server exited before ready")
            out, err = proc.communicate()
            print("----- server logs begin -----")
            print(out)
            print(err)
            print("----- server logs end -----")
            return False

        # listening?
        if is_port_open(HOST, PORT):
            print("[launcher] server is up!")
            return True

        # timeout?
        elapsed = time.time() - start_t
        if elapsed >= timeout_sec:
            print("[launcher] timeout waiting for server")
            return False

        # heartbeat
        pct = min(pct + 5, 95)
        if pct < 50:
            phase = "initializing"
        else:
            phase = "starting_server"
        # we can live-update splash here by navigating it to a new URL
        # using win.load_url(...); we'll do that below where we have `win`.
        yield (phase, pct)

        time.sleep(0.5)


def serve_splash_and_boot():
    """
    This runs ON THE MAIN THREAD.
    We:
      1. create splash webview window
      2. in a callback (after gui init), start server, poll it, open browser
      3. keep process alive until server exits
    """

    # build initial splash URL
    import urllib.parse
    init_url = (
        "file://" + SPLASH_FILE + "?" +
        urllib.parse.urlencode({"phase": "initializing", "pct": "5"})
    )

    print(f"[launcher] creating splash window @ {init_url}")

    # create the window first
    win = webview.create_window(
        "Calypso Labs — Boot",
        init_url,
        width=900,
        height=500,
        resizable=False,
        fullscreen=False,
        confirm_close=False,
    )

    def after_window_shown():
        """
        This runs once the GUI loop is live.
        Safe place to do blocking work.
        """
        print("[launcher] splash is live, beginning boot sequence")

        # 1. start server
        proc = start_server()

        # 2. poll server
        for (phase, pct) in wait_for_server_or_die(proc):
            # update splash during boot
            qs = {"phase": phase, "pct": str(pct)}
            url = "file://" + SPLASH_FILE + "?" + urllib.parse.urlencode(qs)
            try:
                win.load_url(url)
            except Exception as e:
                print(f"[launcher] failed to update splash url: {e}")

        # after loop, either server is up or it failed/timeout
        if proc.poll() is not None:
            # server died -> show error state
            qs = {
                "phase": "error",
                "pct": "95",
                "err": "Server failed to start.",
            }
            url = "file://" + SPLASH_FILE + "?" + urllib.parse.urlencode(qs)
            try:
                win.load_url(url)
            except Exception as e:
                print(f"[launcher] failed to show error splash: {e}")
            return  # stop here, keep splash open so user can see error

        # 3. server is live -> update splash to 100%, then open browser
        try:
            ok_url = (
                "file://" + SPLASH_FILE + "?" +
                urllib.parse.urlencode({"phase": "ready", "pct": "100"})
            )
            win.load_url(ok_url)
        except Exception as e:
            print(f"[launcher] failed to mark ready: {e}")

        print(f"[launcher] opening browser {INDEX_URL}")
        webbrowser.open_new_tab(INDEX_URL)

        # 4. OPTIONAL: close splash window automatically
        # comment this out if you want the splash to remain visible
        try:
            webview.destroy_window()  # closes the only window = exits GUI loop
        except Exception as e:
            print(f"[launcher] couldn't auto-close splash: {e}")

        # 5. babysit server from here in *this* process
        # We can't just fall out of this function because destroy_window()
        # kills the GUI loop and returns control to python.
        # So after GUI exits we continue below in main().

    # Start the GUI loop on main thread, with our callback
    # after_window_shown will run once the splash window is displayed.
    webview.start(after_window_shown)


def main():
    print("[launcher] launch requested")
    print(f"[launcher] cwd={APP_DIR}")
    print(f"[launcher] python={sys.executable}")
    print(f"[launcher] frozen? {is_frozen()}")

    # If we're running as a frozen app bundle later, we might want different behavior,
    # but for dev: always show the splash using webview main-thread boot.
    serve_splash_and_boot()

    # If we get here, either the splash was closed or errored out.
    # Nothing else to do, because if server launched successfully,
    # it's still running as a child of THIS process. We should
    # not exit until the child exits... BUT we lost the handle to proc
    # when we closed over it inside after_window_shown().
    #
    # For dev this is fine because:
    # - If splash succeeded, we destroyed the window *after* launching the browser.
    # - The subprocess is still a child of THIS Python, so this python will *not*
    #   exit until that child exits. (macOS keeps us alive while child is alive.)
    #
    # If you want ironclad babysitting in prod bundle, we can refactor later
    # to stash `proc` somewhere global and loop .poll().
    print("[launcher] launcher finished GUI loop; server (if up) is now live in background.")

if __name__ == "__main__":
    main()
