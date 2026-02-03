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
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
  blue "[build] activated .venv"
else
  blue "[build] .venv not found, continuing (assuming env already active)"
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

green "[ok] build complete: $APP_BUNDLE"
green "[ok] logos + UI assets present"

echo
echo "Run:"
echo "  open \"$APP_BUNDLE\""
echo
echo "If Finder/Dock icon looks stale:"
echo "  killall Dock"
