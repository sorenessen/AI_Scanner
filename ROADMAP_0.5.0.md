# CopyCat v0.5.0 – UX Redesign Roadmap  
Branch: `feat/0.5.0-ux-redesign-experimental`

Goal: Reduce visual “busy-ness” and cognitive load while keeping the full power of CopyCat.  
Strategy: Introduce collapsible layouts, contextual flows, and compact modes without breaking the existing mental model.

---

## 1. Scope & Principles

**Primary objectives**

1. Make the UI feel less overwhelming on first load.
2. Let users focus on *one job at a time* (scan, verify, compare, review).
3. Preserve all existing functionality; this is a **presentation-layer** change.

**Guiding principles**

- **Progressive disclosure** – show the basics first, reveal depth on demand.
- **No dead ends** – every view should have a clear “next thing to do”.
- **Safe to roll back** – keep changes mostly in layout + CSS + small JS glue.

---

## 2. Milestones

- **0.5.0-alpha.1 – Collapsible Shell**
  - Implement collapsible cards on the main page.
  - Basic “focus mode” for the Result card.
  - No new flows yet; everything still behaves like v0.4.x.

- **0.5.0-alpha.2 – Contextual Flows**
  - Introduce student/teacher “flow presets”.
  - Light restructuring of panels (what shows by default for each flow).
  - Add a quick way to toggle flows without reloading.

- **0.5.0-beta – Compact Modes**
  - Add density / compact controls (e.g., “Comfortable / Compact / Minimal”).
  - Tighten typography and spacing based on density.
  - Polish visual hierarchy, keyboard focus, and responsive behavior.

- **0.5.0-rc & Final**
  - Bug fixes, performance checks, copy polish.
  - Update docs, screenshots, and marketing copy.
  - Merge plan + cleanup of experimental flags.

---

## 3. Milestone Details

### 3.1 0.5.0-alpha.1 – Collapsible Cards & Focus Mode

**Goal:** Reduce simultaneous noise by letting users collapse regions and “zoom into” result reading.

**Tasks**

1. **Card-level collapsible controls**
   - Add a small toggle (chevron / “Hide”) to:
     - PD fingerprints / upper-right LCARS panel.
     - Safety / mission log region.
     - Possibly the “What this means” explanation area.
   - Persist collapsed/expanded state in `localStorage` (per card).

2. **Result focus mode**
   - Add a “Focus Result” button near the main Result header.
   - When active:
     - Dim or partially fade other cards (CSS class like `.is-dimmed`).
     - Optionally bump base font-size for the Result area slightly.
   - Keyboard shortcut (e.g., `F`) to toggle focus mode.

3. **Live verification simplification (light pass)**
   - Ensure the Live drawer respects the same typography scale.
   - Optional: hide secondary metrics behind a small “more detail” toggle.

4. **Tech notes**
   - Keep all logic under a feature flag: `window.COPYCAT_UX_050 = true;`
   - No changes to backend APIs.

**Exit Criteria**

- All main cards can be collapsed/expanded without breaking layout.
- Focus mode works and feels stable across themes + scales.
- No JS console errors in common flows.

---

### 3.2 0.5.0-alpha.2 – Contextual Flows (Student vs Teacher vs Power User)

**Goal:** Tailor the experience so each persona sees the panels that matter most first.

**Flows**

- **Student mode**
  - Emphasis: simple verdict, key style hints, and live verification.
  - Default visible:
    - Result card
    - “Writing Tutor” / “Tips”
    - Live verification drawer
  - Advanced stuff (drift, fingerprint, mission log) collapsed by default.

- **Teacher mode**
  - Emphasis: evidence and patterns over time.
  - Default visible:
    - Result
    - Drift / consistency diagnostics
    - History / mission log panels
    - Live verification summary (not necessarily the full drawer open).

- **Power mode (current behavior-ish)**
  - Emphasis: everything at once.
  - All panels expanded (like v0.4.x), but with the improved styling from alpha.1.

**Tasks**

1. **Flow selector**
   - Add a small dropdown or pill group near the top (e.g., “View: Student / Teacher / Power”).
   - Store selection in `localStorage` so the app remembers the last choice.

2. **Flow presets**
   - Map each flow to:
     - Which cards start expanded/collapsed.
     - Which tabs are selected by default in the LCARS side panel.
     - Whether the live drawer auto-opens after scan.

3. **Copy tweaks per flow**
   - Student: friendlier, more guidance (“Here’s how to strengthen this…”).
   - Teacher: more analytical tone (“Patterns across drafts…”).
   - Power: concise, assumes the user already knows the tool.

4. **Telemetry hooks (optional / future)**
   - Light client-side counters (how often each flow is used) that could later be wired to storage.

**Exit Criteria**

- Switching flow presets updates layout instantly, without refreshing.
- Student and Teacher modes feel clearly different in what they show first.
- Power mode does not regress from v0.4.x capabilities.

---

### 3.3 0.5.0-beta – Compact Modes & Visual Hierarchy

**Goal:** Let users control visual density and make the most important information pop first.

**Tasks**

1. **Density control**
   - Add a simple density setting (`Comfortable / Compact / Minimal`) near the UI scale slider.
   - For each density:
     - Adjust vertical margins between rows.
     - Tweak font-size for labels and secondary text.
     - Tighten padding inside cards.

2. **Hierarchy pass**
   - Result card:
     - Make the verdict + donut even more visually dominant.
     - Demote low-priority labels (slightly smaller, muted color).
   - Right panel:
     - Ensure only the active tab feels “lit”; others are subdued.
   - Live drawer:
     - Make the match percentage and timer the first things your eye lands on.

3. **Responsive pass**
   - Confirm the new collapsible/focus behavior works on:
     - Narrow laptop widths.
     - 13" MacBooks.
     - Zoomed browsers (125–150%).

4. **Accessibility**
   - Re-check color contrast after all new themes/densities.
   - Verify keyboard navigation still makes sense with collapsible sections and tabs.

**Exit Criteria**

- Density settings noticeably change feel without breaking layout.
- At “Compact” or “Minimal,” the app feels less busy while still usable.
- No major visual bugs across themes and scale settings.

---

## 4. Testing & QA

For each milestone:

1. **Smoke tests**
   - Run a few known text samples (classic literature, student essay, obvious AI) and confirm:
     - Scans complete.
     - Verdict, donut, and key metrics look correct.
     - Live verification still works end-to-end.

2. **Regression checkpoints**
   - v0.4.x screenshot comparisons for:
     - Result card content.
     - Live drawer metrics.
     - PD fingerprints / LCARS panel content.

3. **Manual persona check**
   - Student mode:
     - Can I paste text, run a scan, and understand what to do next?
   - Teacher mode:
     - Can I see multiple drafts’ history and drift behavior easily?
   - Power mode:
     - Do I still have all panels and detailed metrics available?

---

## 5. Release Plan

1. Develop on `feat/0.5.0-ux-redesign-experimental`.
2. Tag internal checkpoints:
   - `v0.5.0-alpha.1-ux`
   - `v0.5.0-alpha.2-ux`
   - `v0.5.0-beta`
3. Once stable:
   - Merge into `main`.
   - Tag `v0.5.0`.
   - Update README + screenshots + any public docs to reflect the new UX.

---

## 6. Parking Lot / Future Ideas

- Drag-and-drop `.docx` import for the “Text to scan” field.
- A dedicated “Teacher dashboard” that surfaces drift + history across multiple students.
- “Calm mode” that auto-hides secondary panels until hovered or clicked.
- Tutorials / guided tours that highlight panels the first time a user opens a new flow.

