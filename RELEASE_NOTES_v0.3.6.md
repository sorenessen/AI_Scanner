# 🧩 CopyCat v0.3.6 — v0.3.5

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
- `/version`: returns version, model, device, dtype, mode, ensemble, fingerprint_centroids
- (add any others)

**Deprecated / Removed**
- (list if any)

## 🧾 Meta
**Date:** 2025-11-13  
**Tag:** `v0.3.6`  
**Branch:** `main`  
**Merged Into:** `main` (planned/actual)  
**Commit Range:** `v0.3.5..HEAD`

## 🔍 What’s Changed
- v0.3.6: drift diagnostics API + endpoints (#11) — by Soren Essen [41e850bc]
- formatting — by sorenessen [688987d0]
- v0.3.5: Live Verification UX + guarded export + Explain band/PD badges (#10) — by Soren Essen [3a4f56d4]
- workflow guide - RELEASE_WORKFLOW.md — by sorenessen [4e28b08d]
- Adding release note scripts and templates — by sorenessen [f229191f]
- chore: ignore backup file — by sorenessen [e7ad4f4a]

**Full Changelog:** https://github.com/sorenessen/AI_Scanner/compare/v0.3.5...v0.3.6  
(While drafting): https://github.com/sorenessen/AI_Scanner/compare/v0.3.5...HEAD

## 👥 Contributors
- @sorenessen

## 📦 Assets
- Source code (zip)
- Source code (tar.gz)

### 📜 Notes
> Backward-compat / upgrade notes, if any.
