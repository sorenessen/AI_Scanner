# 🧭 CopyCat Release Workflow

This document standardizes how to version, tag, and publish new releases of CopyCat (AI_Scanner).
It covers feature branching, tagging, release note generation, and publishing to GitHub.

🧩 Branching & Versioning Convention
Type	Example	Purpose
Feature	feat/0.3.5-live-ux-export	Active development branch for a feature or improvement
Docs	docs/0.3.5-live-ux-export	Documentation-only updates
Hotfix	fix/0.3.6-pdf-crash	Urgent patch for a bug or stability issue

Rules:

Always start from main for new work:

git checkout main
git pull
git switch -c feat/x.y.z-description


Increment version numbers in app.py and package.json (if present).

🧾 Pre-Release Checklist

Before tagging or merging, verify:

✅ /version endpoint reports the correct version.

✅ UI Live Verification drawer behavior:

Disabled until first scan

Finalize requires ≥ 60 words

Copy Summary + Download .txt gated until Finalize

✅ Explain panel toggles properly and badges render (Human / Mixed / AI / PD).

✅ No innerHTML of null or similar errors in the browser console.

✅ PDF export and logs write correctly.

🧠 Release Notes & Template

We store two reference files:

RELEASE_TEMPLATE.md – general layout for each release.

make_release.sh – auto-fills the template using git history and metadata.

Generate new release notes:

# Example
./make_release.sh v0.3.6 "Drift Diagnostics + UX Alignment"


This will create:

RELEASE_NOTES_v0.3.6.md


Edit it if needed, commit, and push.

🧩 Tagging & Pushing a New Release

After merging your feature branch into main:

# 1. Update local main
git checkout main
git pull origin main

# 2. Tag the new version
git tag -a v0.3.6 -m "CopyCat v0.3.6 — Drift Diagnostics + UX Alignment"

# 3. Push branch and tag
git push origin main
git push origin v0.3.6


To retag (if it pointed to a feature branch):

git tag -d v0.3.6
git tag -a v0.3.6 -m "CopyCat v0.3.6 — Drift Diagnostics + UX Alignment"
git push --force origin v0.3.6

🚀 Creating the GitHub Release
Using GitHub CLI (recommended)
gh release create v0.3.6 \
  --title "CopyCat v0.3.6 — Drift Diagnostics + UX Alignment" \
  --notes-file RELEASE_NOTES_v0.3.6.md

Or manually via GitHub UI

Go to Releases → Draft a new release.

Select the tag (v0.3.6).

Paste contents of RELEASE_NOTES_v0.3.6.md.

Publish release.

🧩 Post-Release Sanity Checks

Run:

curl -s http://localhost:8080/version | jq .


You should see:

{
  "version": "0.3.6",
  "model": "EleutherAI/gpt-neo-1.3B",
  "device": "mps",
  "dtype": "torch.float16",
  "ensemble": false,
  "secondary_model": null,
  "mode": "Balanced",
  "fingerprint_centroids": 4
}


Confirm:

/healthz returns { "ok": true }

GitHub release page lists the correct version and notes.

📦 Optional Enhancements

Add Artifacts to each release:

gh release upload v0.3.6 dist/copycat-mac.zip dist/copycat-src.tar.gz


Add Automation Alias to make_release.sh:

./make_release.sh v0.3.6 && gh release create v0.3.6 -F RELEASE_NOTES_v0.3.6.md

🧰 Recovery / Cleanup Commands
Situation	Command
Undo local tag	git tag -d v0.3.6
Delete remote tag	git push origin :refs/tags/v0.3.6
Remove local feature branch	git branch -d feat/0.3.5-live-ux-export
Remove remote feature branch	git push origin --delete feat/0.3.5-live-ux-export
✅ Example: v0.3.5 Workflow Summary
# From main
git switch -c feat/0.3.5-live-ux-export

# Build + test
uvicorn app:app --port 8080
# …verify features…

# Commit + push
git add .
git commit -m "v0.3.5: Live Verification UX + Guarded Export"
git push --set-upstream origin feat/0.3.5-live-ux-export

# Open PR → Merge → Tag → Push
git checkout main
git pull
git merge feat/0.3.5-live-ux-export
git tag -a v0.3.5 -m "CopyCat v0.3.5 — Live Verification UX + Guarded Export"
git push && git push --tags


Maintained by: @sorenessen
Last updated: $(date +%Y-%m-%d)