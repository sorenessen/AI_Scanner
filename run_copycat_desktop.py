import os
import sys
import time
import threading

# Make sure local imports work when bundled
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import webview
import uvicorn
import app  # forces PyInstaller to include app.py


def start_server():
    # Keep logs minimal; adjust if you need debugging
    uvicorn.run(app.app, host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    # Start the server in the background
    t = threading.Thread(target=start_server, daemon=True)
    t.start()

    # Give the server a moment to start
    time.sleep(1.2)

    # Open a native window (NOT a browser)
    webview.create_window(
        "CopyCat",
        "http://127.0.0.1:8000",
        width=1100,
        height=760,
        resizable=True,
    )
    webview.start()
