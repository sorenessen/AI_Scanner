## Overview

**Focus:** LLM fingerprint UI + drift diagnostics  
**Range:** `v0.3.7` → `HEAD`  
**Commits:** 1

## Highlights

### UI / UX
- (no user-facing UI changes detected in this release)

### Backend / API
- (no external API changes; internal logic only)

### Docs / Ops
- 📚 fixing release notes generation (`d0580c44`)

## Verification Checklist

- [x] Scan → Result → Explain toggle works (where present)
- [x] Live Verification + Finalize stable (≥60 words)
- [x] Copy Summary + Download .txt gated until Finalize
- [x] /version shows correct fields (version/model/device/dtype/mode/ensemble/fingerprint_centroids)
- [x] No console / backend errors in common flows

<details>
<summary><strong>Technical Details</strong></summary>

- Date: 2025-11-14
- Tag: `v0.3.8`
- Branch: `main`
- Commit range: `v0.3.7..HEAD`

</details>

<details>
<summary><strong>What's Changed (commits)</strong></summary>

- `d0580c44` — fixing release notes generation

</details>

## Contributors

- @sorenessen
