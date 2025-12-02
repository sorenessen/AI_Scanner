## Overview

**Focus:** LCARS them + Live Verification + CopyCat voice UI  
**Range:** `HEAD~15` → `HEAD`  
**Commits:** 15

## Highlights

### UI / UX
- 🎨 optional spacer in css padding from top of screen (`a514227e`)
- 🎨 button animations added (`c4794f81`)
- 🎨 restored live verification timer and token counter (`020f8f2f`)
- 🎨 fixed right side tooltips (`8c722444`)

### Backend / API
- (no external API changes; internal logic only)

### Docs / Ops
- (no docs / ops changes recorded for this range)

### Other
- 🧩 info box fix (`fa983526`)
- 🧩 setup safety connection in LCARS tab (`5fc0a17d`)
- 🧩 wired in tab info (`0a2f73b2`)
- 🧩 retrieved JS functionality (`16779445`)
- 🧩 lower right panel added (`34f73168`)
- 🧩 tightened up the results card (`0eedc07d`)
- 🧩 finishing donut guage (`7f19d6a9`)
- 🧩 star trek skin applied (`3bf809a6`)
- 🧩 cleaning duplicates (`248eda06`)
- 🧩 added LCARS theme (`246e7f98`)
- 🧩 made unavailable message verbose (`2ff96290`)

## Verification Checklist

- [x] Scan → Result → Explain toggle works (where present)
- [x] Live Verification + Finalize stable (≥60 words)
- [x] Copy Summary + Download .txt gated until Finalize
- [x] /version shows correct fields (version/model/device/dtype/mode/ensemble/fingerprint_centroids)
- [x] No console / backend errors in common flows

<details>
<summary><strong>Technical Details</strong></summary>

- Date: 2025-12-02
- Tag: `v0.4.0`
- Branch: `main`
- Commit range: `HEAD~15..HEAD`

</details>

<details>
<summary><strong>What's Changed (commits)</strong></summary>

- `fa983526` — info box fix
- `5fc0a17d` — setup safety connection in LCARS tab
- `a514227e` — optional spacer in css padding from top of screen
- `0a2f73b2` — wired in tab info
- `c4794f81` — button animations added
- `020f8f2f` — restored live verification timer and token counter
- `16779445` — retrieved JS functionality
- `34f73168` — lower right panel added
- `0eedc07d` — tightened up the results card
- `7f19d6a9` — finishing donut guage
- `3bf809a6` — star trek skin applied
- `248eda06` — cleaning duplicates
- `8c722444` — fixed right side tooltips
- `246e7f98` — added LCARS theme
- `2ff96290` — made unavailable message verbose

</details>

## Contributors

- @sorenessen
