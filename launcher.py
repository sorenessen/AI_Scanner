import os
import threading
import time
import subprocess
import requests
from pathlib import Path
import webview

# ----------------------------
# CONFIG
# ----------------------------
APP_PORT = 8080
APP_URL = f"http://127.0.0.1:{APP_PORT}"
SPLASH_FILE = Path(__file__).parent / "splash.html"
INDEX_FILE = Path(__file__).parent / "index.html"

# ----------------------------
# STATE
# ----------------------------
server_process = None


def start_server():
    """
    Launch uvicorn in the background.
    We do NOT use --reload here because this is the packaged / end-user experience.
    For dev you can still run uvicorn manually yourself with --reload.
    """
    global server_process
    try:
        server_process = subprocess.Popen(
            [
                "python3", "-m", "uvicorn", "app:app",
                "--port", str(APP_PORT)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # We do NOT read stdout/stderr here (would block). In production
        # you'd redirect to a log file instead.
    except Exception as e:
        print(f"[LAUNCHER ERROR] Could not start server: {e}")
        return False
    return True


def wait_for_server(timeout_sec=30):
    """
    Poll the FastAPI server until it responds 200 OK or we time out.
    Returns True if ready, False if not.
    """
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            r = requests.get(APP_URL)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def boot_sequence(window):
    """
    Runs on a background thread.
    1. tell splash "starting_server" 40%
    2. start uvicorn
    3. poll server
    4. either load index.html (success) or show error state (fail)
    """

    # step 1: starting server
    try:
        window.load_url(
            SPLASH_FILE.as_uri() + "?phase=starting_server&pct=40"
        )
    except Exception as e:
        print(f"[LAUNCHER WARN] splash update failed (starting_server): {e}")

    # step 2: boot uvicorn
    ok = start_server()
    if not ok:
        fail(window, "Could not start local server process.")
        return

    # step 3: wait until server responds
    ready = wait_for_server(timeout_sec=30)

    if ready:
        # tell splash "ready 100%"
        try:
            window.load_url(
                SPLASH_FILE.as_uri() + "?phase=ready&pct=100"
            )
        except Exception as e:
            print(f"[LAUNCHER WARN] splash update failed (ready): {e}")

        # tiny pause so user can read "Complete…"
        time.sleep(0.8)

        # step 4: repoint SAME window at index.html
        try:
            window.load_url(INDEX_FILE.as_uri())
        except Exception as e:
            print(f"[LAUNCHER ERROR] failed to load index.html: {e}")
            fail(window, "Failed to load UI.")
    else:
        fail(window, "Server did not respond in time.")


def fail(window, message):
    """
    Put splash into error state instead of hard-crashing or looping.
    Does NOT spawn new windows. Does NOT recurse. CPU safe.
    """
    safe_msg = message.replace(" ", "%20")
    try:
        window.load_url(
            SPLASH_FILE.as_uri()
            + f"?phase=error&pct=90&err={safe_msg}"
        )
    except Exception as e:
        print(f"[LAUNCHER FATAL] Could not show error screen: {e}")


def main():
    # 1. create splash window, pointing at phase=initializing
    splash_url = SPLASH_FILE.as_uri() + "?phase=initializing&pct=10"
    splash_window = webview.create_window(
        "Calypso Labs — Boot",
        splash_url,
        width=900,
        height=520,
        resizable=False,
    )

    # 2. kick off background boot logic
    t = threading.Thread(
        target=boot_sequence,
        args=(splash_window,),
        daemon=True
    )
    t.start()

    # 3. start UI loop, mac backend = cocoa (no qtpy dependency)
    webview.start(gui="cocoa", debug=False)


if __name__ == "__main__":
    main()
