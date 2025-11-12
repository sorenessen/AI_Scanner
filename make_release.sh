#!/usr/bin/env bash
# make_release.sh — Generate release notes from a template.
# Usage:
#   ./make_release.sh v0.3.6 "Drift Diagnostics + UX Alignment"
#   ./make_release.sh v0.3.6                       # (no codename)
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <version> [codename]" >&2
  exit 1
fi

VERSION="$1"                         # e.g. v0.3.6
CODENAME="${2:-}"                    # optional
DATE="$(date -u +%Y-%m-%d)"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

# Find previous tag (the most recent tag before HEAD). If none, fall back to first commit.
if git describe --tags --abbrev=0 >/dev/null 2>&1; then
  PREV_TAG="$(git describe --tags --abbrev=0 2>/dev/null || true)"
else
  PREV_TAG=""
fi
if [[ -z "${PREV_TAG}" ]]; then
  FIRST_COMMIT="$(git rev-list --max-parents=0 HEAD | tail -n1)"
  PREV_RANGE="$FIRST_COMMIT"
else
  PREV_RANGE="$PREV_TAG"
fi

# Detect GitHub HTTPS URL (for compare links). Supports ssh and https remotes.
REMOTE_URL="$(git config --get remote.origin.url || true)"
REPO_HTTP=""
if [[ "$REMOTE_URL" =~ ^git@github\.com:(.+)\.git$ ]]; then
  REPO_HTTP="https://github.com/${BASH_REMATCH[1]}"
elif [[ "$REMOTE_URL" =~ ^https://github\.com/(.+)\.git$ ]]; then
  REPO_HTTP="https://github.com/${BASH_REMATCH[1]}"
elif [[ "$REMOTE_URL" =~ ^https://github\.com/(.+)$ ]]; then
  REPO_HTTP="https://github.com/${BASHREMATCH[1]}"
fi

# Build "What's Changed" from git log between prev tag (or first commit) and HEAD.
WHATS_CHANGED="$(git log --pretty=format:'- %s — by %an [%h]' "${PREV_RANGE}..HEAD" || true)"

# Compute compare links (only if we know repo URL).
COMPARE_LINK_HEAD=""
COMPARE_LINK_TAG=""
if [[ -n "$REPO_HTTP" ]]; then
  COMPARE_LINK_HEAD="${REPO_HTTP}/compare/${PREV_TAG:-HEAD~1}...HEAD"
  COMPARE_LINK_TAG="${REPO_HTTP}/compare/${PREV_TAG:-HEAD~1}...${VERSION}"
fi

OUT_FILE="RELEASE_NOTES_${VERSION}.md"

# Write file
cat > "$OUT_FILE" <<EOF
# 🧩 CopyCat ${VERSION}${CODENAME:+ — ${CODENAME}}

## 🧭 Overview
> Brief summary (2–3 sentences) of what this release focuses on — major UX or backend goals, user impact, and stability outcomes.

## ✨ Highlights

### 🖥️ UI / UX
- **[Feature]:** What changed and why it matters.
- **[Feature]:** …

### ⚙️ Backend / API
- **[Endpoint/Module]:** Summary of change.
- **[Logic/Perf]:** Safeguards / calibration / optimizations.

### 🧾 Docs / Ops
- **CHANGELOG / README:** Updated.
- **Runtime Config:** New/changed env vars.

## 🧪 Verification Checklist
✅ Scan → Result → Explain toggle works  
✅ Live Verification + Finalize stable (≥60 words)  
✅ Copy Summary + Download .txt gated until Finalize  
✅ /version shows correct fields (version/model/device/dtype/mode/ensemble/fingerprint centroids)  
✅ No console/backend errors  

## 🧩 Technical Details
**Endpoints Updated / Added**
- \`/version\`: returns version, model, device, dtype, mode, ensemble, fingerprint_centroids
- (add any others)

**Deprecated / Removed**
- (list if any)

## 🧾 Meta
**Date:** ${DATE}  
**Tag:** \`${VERSION}\`  
**Branch:** \`${BRANCH}\`  
**Merged Into:** \`main\` (planned/actual)  
**Commit Range:** \`${PREV_RANGE}..HEAD\`

## 🔍 What’s Changed
${WHATS_CHANGED:-_No commit summaries found._}

**Full Changelog:** ${COMPARE_LINK_TAG:-_set after tagging_}  
(While drafting): ${COMPARE_LINK_HEAD:-_no repo URL detected_}

## 👥 Contributors
- @sorenessen

## 📦 Assets
- Source code (zip)
- Source code (tar.gz)

### 📜 Notes
> Backward-compat / upgrade notes, if any.
EOF

echo "✅ Wrote ${OUT_FILE}"
echo
echo "Next steps:"
echo "  1) Review & edit:  ${OUT_FILE}"
echo "  2) Commit:         git add ${OUT_FILE} && git commit -m \"docs(${VERSION}): release notes\""
echo "  3) Tag:            git tag -a ${VERSION} -m \"CopyCat ${VERSION}${CODENAME:+ — ${CODENAME}}\""
echo "  4) Push:           git push && git push --tags"
echo "  5) GitHub Release: paste ${OUT_FILE} into the release body (or upload as an asset)."
