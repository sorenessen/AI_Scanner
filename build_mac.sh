#!/usr/bin/env bash
set -euo pipefail

APP_NAME="CopyCat"
ENTRYPOINT="app.py"

STATIC_DIR="static"
ASSETS_DIR="assets"

ICON_ICNS="${STATIC_DIR}/CopyCat.icns"

# notarization
KEYCHAIN_PROFILE="CopyCat"
TEAM_ID="282VJ37HVD"
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

blue "[preflight] checking required files..."
need "${ENTRYPOINT}"
need "index.html"              # ROOT INDEX (your reality)
need "${STATIC_DIR}"
need "${ICON_ICNS}"

# venv
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
  blue "[preflight] activated .venv"
else
  red "[error] .venv not found."
  exit 1
fi

PYVER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
blue "[preflight] python: ${PYVER}"
if [[ "${PYVER}" != "3.11" && "${PYVER}" != "3.12" && "${PYVER}" != "3.13" ]]; then
  red "[error] expected Python 3.11/3.12/3.13 in .venv (found ${PYVER})"
  exit 1
fi

blue "[clean] removing old artifacts..."
rm -rf build dist dmg_stage
rm -f "${APP_NAME}-mac-arm64.dmg"

blue "[build] PyInstaller..."
# CRITICAL: all flags BEFORE the entrypoint, and bundle ROOT index.html to Resources/index.html
python -m PyInstaller --noconfirm --clean \
  --windowed \
  --name "${APP_NAME}" \
  --icon "${ICON_ICNS}" \
  --collect-submodules numpy \
  --collect-binaries numpy \
  --collect-data numpy \
  --collect-all torch \
  --add-data "index.html:." \
  --add-data "${STATIC_DIR}:${STATIC_DIR}" \
  $( [[ -d "${ASSETS_DIR}" ]] && echo "--add-data ${ASSETS_DIR}:${ASSETS_DIR}" ) \
  "${ENTRYPOINT}"

APP_BUNDLE="dist/${APP_NAME}.app"
need "${APP_BUNDLE}"

RES="${APP_BUNDLE}/Contents/Resources"
PLIST="${APP_BUNDLE}/Contents/Info.plist"

blue "[verify] verify bundled UI exists inside app bundle..."
need "${RES}/index.html"                # ROOT index bundled to Resources/index.html
need "${RES}/${STATIC_DIR}"             # static folder exists

# Optional sanity check to prove the bundle contains the "new UI"
if grep -q "Upload" "${RES}/index.html"; then
  green "[ok] UI check: found 'Upload' in bundled Resources/index.html"
else
  blue "[warn] UI check: did NOT find 'Upload' in bundled Resources/index.html"
  blue "       If browser shows it but app doesn't, your local index.html may differ from what you built."
fi

blue "[bundle] force Info.plist icon keys..."
cp -f "${ICON_ICNS}" "${RES}/CopyCat.icns"
/usr/libexec/PlistBuddy -c "Set :CFBundleIconFile CopyCat.icns" "${PLIST}" 2>/dev/null \
|| /usr/libexec/PlistBuddy -c "Add :CFBundleIconFile string CopyCat.icns" "${PLIST}"

/usr/libexec/PlistBuddy -c "Set :CFBundleIdentifier com.calypso-labs.copycat" "${PLIST}" 2>/dev/null \
|| /usr/libexec/PlistBuddy -c "Add :CFBundleIdentifier string com.calypso-labs.copycat" "${PLIST}"

touch "${APP_BUNDLE}" "${PLIST}" "${RES}/CopyCat.icns"
green "[ok] app build complete: ${APP_BUNDLE}"

# ---- SIGN
if [[ -z "${SIGN_ID}" ]]; then
  red "[error] SIGN_ID is empty."
  echo "Run: security find-identity -v -p codesigning"
  echo "Then: SIGN_ID=\"Developer ID Application: ... (${TEAM_ID})\" ./build_mac.sh"
  exit 1
fi

blue "[sign] signing app bundle..."
if [[ -d "${APP_BUNDLE}/Contents/Frameworks" ]]; then
  codesign --force --options runtime --timestamp --sign "${SIGN_ID}" "${APP_BUNDLE}/Contents/Frameworks"/* 2>/dev/null || true
fi
codesign --force --deep --options runtime --timestamp --sign "${SIGN_ID}" "${APP_BUNDLE}"

blue "[sign] verify codesign..."
codesign --verify --deep --strict --verbose=4 "${APP_BUNDLE}"
spctl --assess --type execute --verbose=4 "${APP_BUNDLE}" || true
green "[ok] signed + verified: ${APP_BUNDLE}"

# ---- DMG
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
blue "[notary] submit + wait..."
xcrun notarytool submit "${DMG_OUT}" \
  --keychain-profile "${KEYCHAIN_PROFILE}" \
  --team-id "${TEAM_ID}" \
  --wait

blue "[notary] staple..."
xcrun stapler staple "${DMG_OUT}"

green "[ok] notarized + stapled: ${DMG_OUT}"

echo
echo "Verification:"
echo "  spctl -a -vv --type open \"${DMG_OUT}\""
echo "  hdiutil attach \"${DMG_OUT}\""
echo "  spctl --assess --type execute --verbose=4 \"/Volumes/${APP_NAME}/${APP_NAME}.app\""
echo "  codesign --verify --deep --strict --verbose=4 \"/Volumes/${APP_NAME}/${APP_NAME}.app\""
echo "  hdiutil detach \"/Volumes/${APP_NAME}\""
