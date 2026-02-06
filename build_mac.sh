#!/usr/bin/env bash
set -euo pipefail

APP_NAME="CopyCat"
ENTRYPOINT="app.py"

# ---- Paths (adjust if you want, but these match your current tree)
STATIC_DIR="static"

# Preferred icon file for macOS bundles (already working for you)
ICON_ICNS="${STATIC_DIR}/CopyCat.icns"

# Logos used by the report UI (adjust names if needed)
COPYCAT_LOGO="${STATIC_DIR}/cat-logo.png"
CALYPSO_LOGO="${STATIC_DIR}/calypso_logo.png"   # <- if you don’t have this, set to an existing file or delete the checks below

# index.html can be at repo root or inside static/
UI_HTML=""
if [[ -f "index.html" ]]; then UI_HTML="index.html"; fi
if [[ -z "${UI_HTML}" && -f "${STATIC_DIR}/index.html" ]]; then UI_HTML="${STATIC_DIR}/index.html"; fi

# ---- Signing / notarization config
KEYCHAIN_PROFILE="CopyCat"      # you provided this
TEAM_ID="282VJ37HVD"            # you provided this

# Set this to your actual identity string from: security find-identity -v -p codesigning
SIGN_ID="${SIGN_ID:-}"

red()   { printf "\033[31m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }
blue()  { printf "\033[34m%s\033[0m\n" "$*"; }

need() {
  if [[ ! -e "$1" ]]; then
    red "[error] missing required path: $1"
    exit 1
  fi
}

blue "[build] preflight..."
need "$ENTRYPOINT"
need "$STATIC_DIR"
need "$ICON_ICNS"
need "$COPYCAT_LOGO"
# CALYPSO_LOGO is optional if you don’t use it
if [[ -e "$CALYPSO_LOGO" ]]; then
  blue "[build] found calypso logo: $CALYPSO_LOGO"
else
  blue "[build] calypso logo not found (ok if you don’t embed it): $CALYPSO_LOGO"
fi

if [[ -z "${UI_HTML}" ]]; then
  red "[error] could not find index.html (repo root or static/index.html)"
  exit 1
fi
blue "[build] using UI html: ${UI_HTML}"

# venv
if [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
  blue "[build] activated .venv"
else
  red "[error] .venv not found. Create it first."
  exit 1
fi

PYVER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${PYVER}" != "3.11" && "${PYVER}" != "3.12" && "${PYVER}" != "3.13" ]]; then
  red "[error] expected Python 3.11/3.12/3.13 in .venv (found ${PYVER})"
  exit 1
fi

blue "[build] clean build/dist..."
rm -rf build dist dmg_stage "${APP_NAME}-mac-arm64.dmg"

blue "[build] PyInstaller..."
python -m PyInstaller --noconfirm --clean \
  --windowed \
  --name "${APP_NAME}" \
  --icon "${ICON_ICNS}" \
  --collect-submodules numpy \
  --collect-binaries numpy \
  --collect-data numpy \
  --collect-all torch \
  --add-data "${UI_HTML}:." \
  --add-data "${STATIC_DIR}:${STATIC_DIR}" \
  "${ENTRYPOINT}" \
  --add-data "assets/copycat_logo.png:assets" \
  --add-data "assets/calypso_logo.png:assets"

APP_BUNDLE="dist/${APP_NAME}.app"
need "${APP_BUNDLE}"

blue "[build] verify bundled UI + assets..."
need "${APP_BUNDLE}/Contents/Resources/$(basename "${UI_HTML}")"
need "${APP_BUNDLE}/Contents/Resources/${STATIC_DIR}/$(basename "${COPYCAT_LOGO}")"
# optional
if [[ -e "$CALYPSO_LOGO" ]]; then
  need "${APP_BUNDLE}/Contents/Resources/${STATIC_DIR}/$(basename "${CALYPSO_LOGO}")"
fi

blue "[build] force Info.plist icon keys..."
PLIST="${APP_BUNDLE}/Contents/Info.plist"
RES="${APP_BUNDLE}/Contents/Resources"

cp -f "${ICON_ICNS}" "${RES}/CopyCat.icns"
/usr/libexec/PlistBuddy -c "Set :CFBundleIconFile CopyCat.icns" "${PLIST}" 2>/dev/null \
|| /usr/libexec/PlistBuddy -c "Add :CFBundleIconFile string CopyCat.icns" "${PLIST}"

# keep bundle id stable (helps caching + logging path stability)
 /usr/libexec/PlistBuddy -c "Set :CFBundleIdentifier com.calypso-labs.copycat" "${PLIST}" 2>/dev/null \
|| /usr/libexec/PlistBuddy -c "Add :CFBundleIdentifier string com.calypso-labs.copycat" "${PLIST}"

touch "${APP_BUNDLE}" "${PLIST}" "${RES}/CopyCat.icns"

green "[ok] app build complete: ${APP_BUNDLE}"

# ---- SIGN
if [[ -z "${SIGN_ID}" ]]; then
  red "[error] SIGN_ID is empty."
  echo "Run this and pick your Developer ID Application identity:"
  echo "  security find-identity -v -p codesigning"
  echo
  echo "Then rerun like:"
  echo "  SIGN_ID=\"Developer ID Application: Your Name (${TEAM_ID})\" ./build_mac.sh"
  exit 1
fi

blue "[sign] signing app bundle..."
# sign nested libs first
codesign --force --options runtime --timestamp --sign "${SIGN_ID}" "${APP_BUNDLE}/Contents/Frameworks"/* 2>/dev/null || true
# sign the app itself
codesign --force --deep --options runtime --timestamp --sign "${SIGN_ID}" "${APP_BUNDLE}"

blue "[sign] verify codesign..."
codesign --verify --deep --strict --verbose=4 "${APP_BUNDLE}"
spctl --assess --type execute --verbose=4 "${APP_BUNDLE}" || true

green "[ok] signed + verified: ${APP_BUNDLE}"

# ---- DMG (stage folder avoids /Volumes confusion)
blue "[dmg] staging..."
mkdir -p dmg_stage
ditto "${APP_BUNDLE}" "dmg_stage/${APP_NAME}.app"

DMG_OUT="${APP_NAME}-mac-arm64.dmg"
blue "[dmg] creating ${DMG_OUT}..."
hdiutil create -volname "${APP_NAME}" \
  -srcfolder "dmg_stage" \
  -ov -format UDZO \
  "${DMG_OUT}"

green "[ok] dmg created: ${DMG_OUT}"

# ---- NOTARIZE + STAPLE
blue "[notary] submit..."
xcrun notarytool submit "${DMG_OUT}" \
  --keychain-profile "${KEYCHAIN_PROFILE}" \
  --team-id "${TEAM_ID}" \
  --wait

blue "[notary] staple..."
xcrun stapler staple "${DMG_OUT}"

green "[ok] notarized + stapled: ${DMG_OUT}"

echo
echo "Final verify (mount + assess the .app inside):"
echo "  hdiutil attach \"${DMG_OUT}\""
echo "  spctl --assess --type execute --verbose=4 \"/Volumes/${APP_NAME}/${APP_NAME}.app\""
echo "  hdiutil detach \"/Volumes/${APP_NAME}\""
