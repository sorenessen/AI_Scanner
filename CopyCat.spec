# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all, collect_submodules

# --- Collect heavy libs that PyInstaller often misses on macOS ---
torch_datas, torch_binaries, torch_hidden = collect_all("torch")

# Optional but commonly needed with torch stacks (safe even if unused)
extra_hidden = []
for m in ("torchvision", "torchaudio", "transformers"):
    try:
        extra_hidden += collect_submodules(m)
    except Exception:
        pass

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=torch_binaries,
    datas=torch_datas,
    hiddenimports=torch_hidden + extra_hidden,
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
    name='CopyCat',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch="x86_64",   # force intel build output
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
    name='CopyCat',
)

app = BUNDLE(
    coll,
    name='CopyCat.app',
    icon=None,
    bundle_identifier=None,
)
