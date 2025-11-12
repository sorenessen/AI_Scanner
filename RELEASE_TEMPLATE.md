# 🧩 CopyCat Release Template

## 🚀 Version + Codename
`CopyCat vX.X.X — <Short Codename>`  
*(Example: “v0.3.6 — Drift Diagnostics + UX Alignment”)*

---

## 🧭 Overview
> Brief summary (2–3 sentences) of what this release focuses on — major UX or backend goals, user impact, and stability outcomes.

---

## ✨ Highlights

### 🖥️ UI / UX
- **[Feature Name]:** Describe what changed and why it matters.  
- **[Feature Name]:** Another highlight with a short explanation.  
- (Add 2–6 bullets total)

### ⚙️ Backend / API
- **[Endpoint or Module]:** Describe what was added or refined.  
- **[Logic]:** Note new safeguards, caps, or calculation changes.  
- **[Performance]:** Mention optimizations or model-handling updates.

### 🧾 Docs / Ops
- **CHANGELOG / README:** Updated version and endpoints list.  
- **Diagrams:** Added or refreshed for clarity.  
- **Runtime Config:** Mention new env vars, feature toggles, or CLI switches.

---

## 🧪 Verification Checklist
✅ Scan → Result → Explain toggle works  
✅ Live Verification and Finalize stable  
✅ Copy Summary + Download .txt gated correctly  
✅ /version shows correct version & mode  
✅ No console or backend errors  

---

## 🧩 Technical Details
**Endpoints Updated / Added**
- `/version`: Now includes `<fields>`
- `/auth/sample/*`: (Describe any flow changes)

**Deprecated / Removed**
- (List if applicable)

---

## 🧾 Meta
**Tag:** `vX.X.X`  
**Branch:** `feat/X.X.X-<short-name>`  
**Merged Into:** `main`  
**Commit Range:** `[hash]...HEAD`  

---

## 🔍 What’s Changed
- `docs(vX.X.X): <short summary>` — by @sorenessen in #<PR>
- (Autofilled by GitHub)

**Full Changelog:** [`v(X-1).X...vX.X.X`](#)

---

## 👥 Contributors
- @sorenessen  
- (Add any others here)

---

## 📦 Assets
- Source code (zip)  
- Source code (tar.gz)

---

### 📜 Notes
> Optional: include backward-compatibility or upgrade notes here (e.g., “Requires new env var,” “/config schema updated,” etc.)

--- 

### Manual Updates for Release Notes - useful commands only if not updating automatically

- cp RELEASE_TEMPLATE.md RELEASE_NOTES_vX.X.X.md

## Edit the Placeholders and Push - EXAMPLE
- git add RELEASE_NOTES_v0.3.6.md
- git commit -m "docs(v0.3.6): release notes"
- git push


