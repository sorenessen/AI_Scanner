# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_all

datas = [('app.py', '.'), ('index.html', '.'), ('splash.html', '.')]
binaries = []
hiddenimports = ['stylometry']

# collected packages (your list + webview; left your duplicates as-is to avoid churn)
for pkg in [
    'fastapi','uvicorn','pydantic','reportlab','transformers','torch','starlette',
    'anyio','websockets','jinja2','uvloop','click','certifi','charset_normalizer',
    'idna','urllib3','packaging','filelock','safetensors','regex','sympy','networkx',
    'tqdm','numpy','tokenizers','huggingface_hub','fsspec','pillow','requests',
    'typing_extensions','starlette','markdown-it-py','mdurl','pygments',
    'importlib_metadata','reportlab.lib','reportlab.graphics','reportlab.pdfgen',
    'reportlab.platypus','reportlab.rl_config','reportlab.pdfbase',
    'reportlab.graphics.charts.barcharts','reportlab.graphics.shapes',
    'reportlab.graphics.widgets.grids','reportlab.graphics.widgets.markers',
    'reportlab.lib.fonts','reportlab.lib.colors','reportlab.lib.pagesizes',
    'webview',   # <-- ADD THIS
]:
    try:
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hiddenimports += h
    except Exception:
        pass

project_dir = os.path.abspath(".")
a = Analysis(
    ['launcher.py'],
    pathex=[project_dir],        # <-- set pathex
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AI_Text_Scanner',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                   # <-- was True; turn OFF on macOS
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,                   # <-- OFF here too
    upx_exclude=[],
    name='AI_Text_Scanner',
)
app = BUNDLE(
    coll,
    name='AI_Text_Scanner.app',
    icon=None,
    bundle_identifier=None,
)
