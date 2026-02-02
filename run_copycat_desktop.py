import os
import sys
import time
import threading
import urllib.request

# Make sure local imports work when bundled
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import webview
import uvicorn
import app  # forces PyInstaller to include app.py

HOST = "127.0.0.1"
PORT = 8000

def start_server():
    # Keep logs minimal; adjust if you need debugging
    uvicorn.run(app.app, host=HOST, port=PORT, log_level="warning")

def wait_for_server(timeout_s: float = 8.0) -> bool:
    deadline = time.time() + timeout_s
    url = f"http://{HOST}:{PORT}/healthz"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=0.35) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.12)
    return False

if __name__ == "__main__":
    # Start the server in the background
    t = threading.Thread(target=start_server, daemon=True)
    t.start()

    ok = wait_for_server()
    # Even if healthz isn't ready, try to open UI; it may still come up.
    target = f"http://{HOST}:{PORT}/ui"

    webview.create_window(
        "CopyCat",
        target,
        width=1100,
        height=760,
        resizable=True,
    )
    webview.start()
