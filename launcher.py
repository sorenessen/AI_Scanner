import time
import threading
import webbrowser
import sys
import socket

# --- bring in your server code directly ---
import uvicorn
from app import app  # this imports FastAPI app from app.py

HOST = "127.0.0.1"
PORT = 8080

def is_port_open(host, port):
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False

def run_server_blocking():
    """
    Run uvicorn in this same process/thread.
    We disable reload so it doesn't try to fork.
    """
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info",
    )

def main():
    print("[launcher] starting embedded server")

    # start uvicorn in a background thread so launcher can continue
    server_thread = threading.Thread(target=run_server_blocking, daemon=True)
    server_thread.start()

    # wait for server to accept connections
    pct = 0
    for _ in range(40):  # ~20 seconds total (40 * 0.5s)
        if is_port_open(HOST, PORT):
            print("[launcher] server is listening")
            break
        time.sleep(0.5)
        pct += 5
        print(f"[launcher] waiting for server... {pct}%")
    else:
        print("[launcher] FATAL: server did not start.")
        # don't open browser
        # keep process alive a bit so you can read any printed uvicorn errors
        time.sleep(5)
        return

    url = f"http://{HOST}:{PORT}/"
    print(f"[launcher] opening browser at {url}")
    webbrowser.open(url, new=1)

    # keep the main thread alive as long as uvicorn thread is alive
    try:
        while server_thread.is_alive():
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[launcher] interrupted, exiting")

if __name__ == "__main__":
    main()
