import os
import sys
import time
import socket
import subprocess
import threading

import webview  # pywebview

HOST = "127.0.0.1"
PORT = 8080
APP_MOD = "app:app"


def is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def is_port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.3):
            return True
    except OSError:
        return False


def server_command():
    """
    Return (cmd, cwd) for launching uvicorn.
    Dev mode:
        [<venv python>, "-m", "uvicorn", "app:app", ...]
        cwd = project dir
    Frozen mode:
        The PyInstaller bundle drops our files in the same dir as sys.executable
        AND it bundles uvicorn. We can still call '<exe> -m uvicorn'.
        But we MUST NOT recurse into launcher.main() again.

        We solve that by setting CALYPSO_CHILD=1 for the child.
        Child checks that env var and skips launching the GUI.
    """
    if is_frozen():
        app_dir = os.path.dirname(sys.executable)
        python_exec = sys.executable
    else:
        app_dir = os.path.dirname(os.path.abspath(__file__))
        python_exec = sys.executable

    cmd = [
        python_exec,
        "-m", "uvicorn",
        APP_MOD,
        "--host", HOST,
        "--port", str(PORT),
    ]
    return cmd, app_dir


def start_server():
    cmd, cwd = server_command()

    env = os.environ.copy()
    # mark child so it won't try to launch GUI logic if we're frozen
    env["CALYPSO_CHILD"] = "1"

    print(f"[launcher] starting server: {cmd} (cwd={cwd})")
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc


def pump_logs(proc):
    def _pump(stream, label):
        for line in iter(stream.readline, ""):
            if not line:
                break
            print(f"[server:{label}] {line.rstrip()}")
    threading.Thread(target=_pump, args=(proc.stdout, "stdout"), daemon=True).start()
    threading.Thread(target=_pump, args=(proc.stderr, "stderr"), daemon=True).start()


def wait_for_server(timeout_sec=15):
    start = time.time()
    pct = 0
    while True:
        if is_port_open(HOST, PORT):
            print("[launcher] server is UP")
            return True
        if time.time() - start > timeout_sec:
            print("[launcher] ERROR: timeout waiting for server")
            return False
        pct += 5
        print(f"[launcher] waiting for server... {pct}%")
        time.sleep(0.3)


def run_gui(app_url: str):
    """
    Create the pywebview window and BLOCK until the user closes it.
    This is critical: we don't want to kill uvicorn early.
    """

    window = webview.create_window(
        title="Calypso Labs — AI Text Scanner",
        url=app_url,
        width=1000,
        height=700,
        resizable=True,
        confirm_close=True,
    )

    # IMPORTANT: webview.start() will block until the window is closed
    # (on Cocoa, when we don't give it a separate gui loop thread).
    print(f"[launcher] showing client window @ {app_url}")
    webview.start(gui="cocoa")
    print("[launcher] window closed")


def main():
    # if we are the CHILD uvicorn process inside frozen mode:
    # bail immediately so we don't recurse into launching uvicorn again.
    if os.environ.get("CALYPSO_CHILD") == "1":
        # We're not supposed to be here in normal run,
        # but just in case, don't spin up GUI again.
        print("[launcher] CALYPSO_CHILD=1 -> child process context, exiting main() early")
        return

    print("[launcher] launch requested")
    mode = "frozen" if is_frozen() else "dev"
    print(f"[launcher] mode = {mode}")

    # 1. start backend
    proc = start_server()
    pump_logs(proc)

    # 2. wait for listen socket
    if not wait_for_server(timeout_sec=20):
        print("[launcher] backend failed, terminating")
        try:
            proc.terminate()
        except Exception:
            pass
        return

    # 3. GUI (blocking)
    app_url = f"http://{HOST}:{PORT}/"
    try:
        run_gui(app_url)
    finally:
        # 4. cleanup server
        print("[launcher] shutting down server process")
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        print("[launcher] server process ended")
        print("[launcher] clean exit")


if __name__ == "__main__":
    # guard against double-launch loops in some weird macOS re-open flows
    if os.environ.get("CALYPSO_LABS_LAUNCHED") == "1":
        print("[launcher] already running, aborting duplicate launch")
        sys.exit(0)
    os.environ["CALYPSO_LABS_LAUNCHED"] = "1"

    main()
