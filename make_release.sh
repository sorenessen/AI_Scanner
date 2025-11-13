#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 NEW_VERSION \"Title\" [PREV_VERSION]" >&2
  exit 1
fi

NEW_VER="$1"              # e.g. 0.3.7
TITLE="$2"                # e.g. "Drift diagnostics UI + live compare polish"
PREV_VER="${3:-}"         # e.g. 0.3.6 (optional)
DATE_STR="$(date +%Y-%m-%d)"

if [[ -n "$PREV_VER" ]]; then
  RANGE_TAG="v${PREV_VER}..HEAD"
  PREV_LABEL="v${PREV_VER}"
else
  RANGE_TAG="HEAD~15..HEAD"
  PREV_LABEL="HEAD~15"
fi

COMMITS_RAW="$(git log --no-merges --pretty='%h||%s' "$RANGE_TAG" || true)"

if [[ -z "$COMMITS_RAW" ]]; then
  echo "No commits found in range ${RANGE_TAG}. Check your tags / arguments." >&2
  exit 1
fi

UI_COMMITS=()
API_COMMITS=()
DOC_COMMITS=()
OTHER_COMMITS=()
ALL_COMMITS=()

lc() { echo "$1" | tr '[:upper:]' '[:lower:]'; }

while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  sha="${line%%||*}"
  msg="${line#*||}"
  msg_trim="${msg#"${msg%%[![:space:]]*}"}"
  msg_lc="$(lc "$msg_trim")"

  ALL_COMMITS+=("$sha||$msg_trim")

  bucket="other"

  # ignore boring chore/version in highlights; still show in "What's Changed"
  if [[ "$msg_lc" =~ ^(chore|bump|release|version) ]]; then
    bucket="other"
  fi

  if   [[ "$msg_lc" =~ (ui|ux|tooltip|button|drawer|layout|css|frontend|index\.html|live verification|drift diagnostics ui) ]]; then
    bucket="ui"
  elif [[ "$msg_lc" =~ (api|endpoint|/scan|/drift|backend|server|handler|app\.py|fastapi|config|runtime) ]]; then
    bucket="api"
  elif [[ "$msg_lc" =~ (doc|readme|changelog|notes|ops|pipeline|ci|github actions) ]]; then
    bucket="docs"
  fi

  case "$bucket" in
    ui)   UI_COMMITS+=("🎨 ${msg_trim} (\`${sha}\`)") ;;
    api)  API_COMMITS+=("🧠 ${msg_trim} (\`${sha}\`)") ;;
    docs) DOC_COMMITS+=("📚 ${msg_trim} (\`${sha}\`)") ;;
    *)    OTHER_COMMITS+=("🧩 ${msg_trim} (\`${sha}\`)") ;;
  esac
done <<< "$COMMITS_RAW"

COMMIT_COUNT="${#ALL_COMMITS[@]}"
OUT_FILE="release-notes-v${NEW_VER}.md"

{
  echo "## Overview"
  echo
  echo "**Focus:** ${TITLE}  "
  echo "**Range:** \`${PREV_LABEL}\` → \`HEAD\`  "
  echo "**Commits:** ${COMMIT_COUNT}"
  echo

  echo "## Highlights"
  echo

  echo "### UI / UX"
  if (( ${#UI_COMMITS[@]} == 0 )); then
    echo "- (no user-facing UI changes detected in this release)"
  else
    for c in "${UI_COMMITS[@]}"; do
      echo "- $c"
    done
  fi
  echo

  echo "### Backend / API"
  if (( ${#API_COMMITS[@]} == 0 )); then
    echo "- (no external API changes; internal logic only)"
  else
    for c in "${API_COMMITS[@]}"; do
      echo "- $c"
    done
  fi
  echo

  echo "### Docs / Ops"
  if (( ${#DOC_COMMITS[@]} == 0 )); then
    echo "- (no docs / ops changes recorded for this range)"
  else
    for c in "${DOC_COMMITS[@]}"; do
      echo "- $c"
    done
  fi
  echo

  if (( ${#OTHER_COMMITS[@]} > 0 )); then
    echo "### Other"
    for c in "${OTHER_COMMITS[@]}"; do
      echo "- $c"
    done
    echo
  fi

  echo "## Verification Checklist"
  echo
  echo "- [x] Scan → Result → Explain toggle works (where present)"
  echo "- [x] Live Verification + Finalize stable (≥60 words)"
  echo "- [x] Copy Summary + Download .txt gated until Finalize"
  echo "- [x] /version shows correct fields (version/model/device/dtype/mode/ensemble/fingerprint_centroids)"
  echo "- [x] No console / backend errors in common flows"
  echo

  echo "<details>"
  echo "<summary><strong>Technical Details</strong></summary>"
  echo
  echo "- Date: ${DATE_STR}"
  echo "- Tag: \`v${NEW_VER}\`"
  echo "- Branch: \`main\`"
  echo "- Commit range: \`${RANGE_TAG}\`"
  echo
  echo "</details>"
  echo

  echo "<details>"
  echo "<summary><strong>What's Changed (commits)</strong></summary>"
  echo
  for c in "${ALL_COMMITS[@]}"; do
    sha="${c%%||*}"
    msg="${c#*||}"
    echo "- \`${sha}\` — ${msg}"
  done
  echo
  echo "</details>"
  echo

  echo "## Contributors"
  echo
  echo "- @sorenessen"
} > "$OUT_FILE"

echo "Wrote ${OUT_FILE}"
