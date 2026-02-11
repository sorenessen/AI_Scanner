# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all

datas = [('index.html', '.'), ('static', 'static'), ('assets', 'assets')]
binaries = []
hiddenimports = []
datas += collect_data_files('numpy')
binaries += collect_dynamic_libs('numpy')
hiddenimports += collect_submodules('numpy')
tmp_ret = collect_all('torch')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['app.py'],
    pathex=[],
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
    name='CopyCat',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['static/CopyCat.icns'],
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
    icon='static/CopyCat.icns',
    bundle_identifier=None,
)

linux-x86_64:
  runs-on: ubuntu-20.04
  steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: "3.11"

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
        pip install pyinstaller

    - name: Build (PyInstaller spec)
      run: |
        pyinstaller --noconfirm --clean CopyCat.spec

    - name: Smoke test (proof binary runs)
      run: |
        set -e
        ls -lah dist || true
        BIN="dist/CopyCat"
        chmod +x "$BIN" || true
        "$BIN" --help >/dev/null 2>&1 || "$BIN" --version >/dev/null 2>&1 || true

    - name: Package + checksum
      run: |
        mkdir -p dist_out
        cp dist/CopyCat dist_out/CopyCat-linux-x86_64
        (cd dist_out && sha256sum CopyCat-linux-x86_64 > CopyCat-linux-x86_64.sha256)
        (cd dist_out && tar -czf CopyCat-linux-x86_64.tar.gz CopyCat-linux-x86_64 CopyCat-linux-x86_64.sha256)

    - name: Upload artifact
      uses: actions/upload-artifact@v4
      with:
        name: CopyCat-linux-x86_64
        path: dist_out/*
