import os
import sys

# Ensure local imports work in both dev + PyInstaller onefile
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import uvicorn
import app  # <-- this forces PyInstaller to bundle app.py

if __name__ == "__main__":
    uvicorn.run(app.app, host="127.0.0.1", port=8000)
