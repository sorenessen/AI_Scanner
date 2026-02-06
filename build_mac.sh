#!/usr/bin/env bash
set -euo pipefail

APP_NAME="CopyCat"
ENTRYPOINT="app.py"

ICON_PATH="assets/CopyCat.icns"
UI_HTML="index.html"
STATIC_DIR="static"
ASSETS_DIR="assets"

# --- helpers
red()   { printf "\033[31m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }
blue()  { printf "\033[34m%s\033[0m\n" "$*"; }

need() {
  if [[ ! -e "$1" ]]; then
    red "[error] missing required path: $1"
    exit 1
  fi
}

# --- preflight
blue "[build] preflight checks..."
need "$ENTRYPOINT"
need "$ICON_PATH"
need "$UI_HTML"
need "$STATIC_DIR"
need "$ASSETS_DIR"
need "$ASSETS_DIR/copycat_logo.png"
need "$ASSETS_DIR/calypso_logo.png"

# --- venv (optional)
if [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
  blue "[build] activated .venv"
else
  red "[error] .venv not found. Create it first."
  exit 1
fi

PYVER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PYVER" != "3.11" ]]; then
  red "[error] build requires Python 3.11.x in .venv (found $PYVER)"
  exit 1
fi


# --- clean
blue "[build] cleaning build/dist..."
rm -rf build dist

# --- build
blue "[build] running PyInstaller..."
python -m PyInstaller --noconfirm --clean \
  --windowed \
  --name "$APP_NAME" \
  --icon "$ICON_PATH" \
  --collect-submodules numpy \
  --collect-binaries numpy \
  --collect-data numpy \
  --hidden-import numpy.core._multiarray_umath \
  --hidden-import numpy._core._multiarray_umath \
  --collect-all torch \
  --add-data "$UI_HTML:." \
  --add-data "$STATIC_DIR:$STATIC_DIR" \
  --add-data "$ASSETS_DIR:$ASSETS_DIR" \
  "$ENTRYPOINT"



# --- verify output
APP_BUNDLE="dist/${APP_NAME}.app"
if [[ ! -d "$APP_BUNDLE" ]]; then
  red "[error] expected app bundle not found: $APP_BUNDLE"
  ls -la dist || true
  exit 1
fi

blue "[build] verifying bundle contents..."
need "$APP_BUNDLE/Contents/Resources/index.html"
need "$APP_BUNDLE/Contents/Resources/assets/copycat_logo.png"
need "$APP_BUNDLE/Contents/Resources/assets/calypso_logo.png"

blue "[build] forcing macOS bundle icon keys..."

PLIST="$APP_BUNDLE/Contents/Info.plist"
RES="$APP_BUNDLE/Contents/Resources"

# Put a known icon file in Resources
cp -f "$ICON_PATH" "$RES/CopyCat.icns"

# Ensure Info.plist points at it (no extension in CFBundleIconFile)
 /usr/libexec/PlistBuddy -c "Set :CFBundleIconFile CopyCat" "$PLIST" 2>/dev/null \
|| /usr/libexec/PlistBuddy -c "Add :CFBundleIconFile string CopyCat" "$PLIST"

# Optional but helps caching a LOT — keep it stable:
 /usr/libexec/PlistBuddy -c "Set :CFBundleIdentifier com.calypso-labs.copycat" "$PLIST" 2>/dev/null \
|| /usr/libexec/PlistBuddy -c "Add :CFBundleIdentifier string com.calypso-labs.copycat" "$PLIST"

# Touch so Finder/Dock notices
touch "$APP_BUNDLE" "$PLIST" "$RES/CopyCat.icns"


green "[ok] build complete: $APP_BUNDLE"
green "[ok] logos + UI assets present"

echo
echo "Run:"
echo "  open \"$APP_BUNDLE\""
echo
echo "If Finder/Dock icon looks stale:"
echo "  killall Dock"
