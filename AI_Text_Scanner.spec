# -*- mode: python ; coding: utf-8 -*-

import sys
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# --- force include runtime modules that app.py needs but launcher.py doesn't import directly ---
hiddenmods = []
hiddenmods += collect_submodules("fastapi")
hiddenmods += collect_submodules("starlette")
hiddenmods += collect_submodules("pydantic")
hiddenmods += collect_submodules("uvicorn")
hiddenmods += collect_submodules("uvloop")
hiddenmods += collect_submodules("anyio")
hiddenmods += collect_submodules("websockets")
hiddenmods += collect_submodules("jinja2")
hiddenmods += collect_submodules("torch")
hiddenmods += collect_submodules("transformers")
hiddenmods += collect_submodules("reportlab")

# de-dupe
hiddenmods = sorted(set(hiddenmods))

# data files some libs need at runtime (templates, fonts, etc.)
datas = []
datas += collect_data_files("fastapi")
datas += collect_data_files("starlette")
datas += collect_data_files("jinja2")
datas += collect_data_files("reportlab")
datas += collect_data_files("transformers")
datas += collect_data_files("torch")

# our own app files that must ship with the bundle
datas += [
    ('app.py', '.'),
    ('index.html', '.'),
    ('splash.html', '.'),
]

a = Analysis(
    ['launcher.py'],
    pathex=[],          # current working dir is fine
    binaries=[],
    datas=datas,
    hiddenimports=hiddenmods,
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
    upx=True,
    console=False,                  # no terminal on launch
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
    upx=True,
    upx_exclude=[],
    name='AI_Text_Scanner',
)

app = BUNDLE(
    coll,
    name='AI_Text_Scanner.app',
    icon=None,
    bundle_identifier=None,
)

