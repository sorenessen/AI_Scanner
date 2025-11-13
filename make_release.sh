#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 3 ]; then
  echo "Usage: $0 VERSION \"Title\" PREV_TAG" >&2
  echo "Example: $0 0.3.6 \"Drift diagnostics API + endpoints\" v0.3.5" >&2
  exit 1
fi

VERSION="$1"          # e.g. 0.3.6
TITLE="$2"            # e.g. "Drift diagnostics API + endpoints"
PREV_TAG="$3"         # e.g. v0.3.5
TAG="v$VERSION"       # v0.3.6
DATE=$(date +%Y-%m-%d)
OUT="RELEASE_NOTES_v$VERSION.md"

# Collect commits between previous tag and HEAD
COMMITS=$(git log --pretty='- %h %s (%an)' "$PREV_TAG"..HEAD)

cat > "$OUT" <<EOF
# CopyCat $TAG — $TITLE

## Overview

Short summary (2–3 sentences) of what this release focuses on: key UX or backend goals, user impact, and stability outcomes.

## Highlights

### UI / UX

- ...

### Backend / API

- ...

### Docs / Ops

- CHANGELOG / README updated.
- Runtime config docs updated as needed.

## Verification Checklist

- [x] Scan → Result → Explain toggle works
- [x] Live Verification + Finalize stable (≥60 words)
- [x] Copy Summary + Download .txt gated until Finalize
- [x] /version shows correct fields (version/model/device/dtype/mode/ensemble/fingerprint_centroids)
- [x] No console/backend errors

## Technical Details

### Endpoints Updated / Added

- ...

### Deprecated / Removed

- _None._  <!-- update if needed -->

## Meta

- Date: $DATE
- Tag: $TAG
- Branch: main
- Commit Range: $PREV_TAG..HEAD

## What's Changed

$COMMITS

## Contributors

- @sorenessen

EOF

echo "Wrote $OUT"
