1️⃣ ROADMAP.md (ready to paste)
# CopyCat Multimodal Forensics – Roadmap

## 0. Vision

CopyCat is evolving from a text-only AI content auditor into a **multimodal forensic suite** that can detect AI involvement in:

- Text
- Images
- Audio
- Video

The goal is to give users **defensible evidence**, not just single-number “AI scores.” Each modality provides an overall verdict with supporting signals and human-readable explanations.

Core customers: educators, researchers, journalists, compliance / legal, and enterprises verifying the authenticity of digital content.

---

## 1. Architecture Overview

### 1.1. Backend

- Framework: FastAPI
- Core concept: **scan job**
- New entities:
  - `scan_jobs` – one row per submitted scan
  - `scan_artifacts` – files or text blobs associated with a job (type: `text | image | audio | video`)
- Object storage for large files (S3-compatible or local disk during development).
- Each modality has its own analyzer module:

  - `text_detector.py`  (existing)
  - `image_detector.py`
  - `audio_detector.py`
  - `video_detector.py`

All analyzers return a common schema:

```json
{
  "job_id": "uuid",
  "modality": "image",
  "overall_verdict": "likely_ai_generated",
  "confidence": 0.87,
  "signals": {
    "...": "..."
  },
  "explanations": [
    "Short human-readable bullet 1",
    "Short human-readable bullet 2"
  ]
}

1.2. Frontend

Existing CopyCat UI extended with a Media tab for:

Image

Audio

Video

Each modality reuses the same pattern:

Upload area

Overall verdict capsule + confidence bar

“Signals” list (chips or tag-like buttons)

“What this means” explanation box

Optional visualizations (e.g., heatmap for images, waveform for audio, timeline stripe for video)

2. Phases
Phase 0 – Platform groundwork

Goals

Introduce modality concept to the text pipeline.

Define the unified API response schema.

Prepare job/artifact storage and logging.

Tasks

 Add modality: "text" to existing /scan responses.

 Introduce ScanJob + ScanArtifact models and DB tables.

 Implement /api/v1/scan/{job_id} to fetch past results.

 Add basic “Media” tab and placeholders in the UI (no real detectors yet).

Exit criteria

Text scans work exactly as before, but now include modality and job_id.

UI has a visible but clearly labeled “coming soon” area for image/audio/video.

Phase 1 – Image authenticity (MVP)

Question answered:

“Is this image likely AI-generated, camera-original, or AI-modified?”

Signals

EXIF / metadata analysis

JPEG quantization / compression profile

AI-vs-real classifier (deep model)

Error Level Analysis (ELA) for edit-spotting

Tasks

 Implement image_detector.py:

 extract_exif()

 analyze_quantization()

 classify_ai_vs_real() (using pre-trained model)

 compute_ela() + summary stats

 combine_image_signals() → (overall_verdict, confidence, signals)

 Add POST /api/v1/scan/image endpoint.

 In /api/v1/scan, auto-detect modality when an image is uploaded and route accordingly.

 Build the Image Authenticity panel in the UI:

 Drag-and-drop upload

 Thumbnail preview

 Verdict + confidence

 “Signals” chips (metadata, compression, AI fingerprint, editing)

 ELA heatmap thumbnail (if editing suspected)

 Log scans for later evaluation.

Exit criteria

Users can upload an image and receive a clear verdict + explanations.

Internal test set of real vs AI images shows reasonable performance.

No regressions to text scanning.

Phase 2 – Audio authenticity

Question answered:

“Does this audio clip contain AI-generated or cloned voice?”

Signals

Spectrogram classifier (AI voice vs human)

Prosody and micro-variation (pitch, jitter, shimmer, speech rate)

Noise and environment profile

Breath / non-speech behavior

Tasks

 Implement audio_detector.py with preprocessing (resample, mono, normalization).

 Train or integrate a spectrogram-based classifier.

 Compute prosody and noise-profile features.

 Expose POST /api/v1/scan/audio.

 UI: Audio upload + basic waveform, verdict + confidence, explanation box.

Exit criteria

Audio clips can be uploaded and analyzed.

Clear documentation of strengths/limitations (e.g., very short clips, heavy music).

Phase 3 – Video / deepfake detection

Question answered:

“Is this video or any part of it likely AI-generated or heavily AI-edited (deepfake)?”

Signals

Frame-level AI/real classification

Face landmarks & blink / motion irregularities

Optical flow anomalies

Audio-video sync (if audio present)

Tasks

 Implement video_detector.py:

 Frame extraction (e.g., 3–5 fps).

 Per-frame AI probabilities using image classifier.

 Optional face-landmark analysis and blink rate.

 Optical flow analysis for motion irregularities.

 Aggregate per-segment verdicts.

 Expose POST /api/v1/scan/video.

 UI: upload, basic player, timeline stripe showing suspicion level over time.

Exit criteria

Videos can be analyzed; suspicious segments are highlighted.

Performance is acceptable on target hardware / instance types.

Phase 4 – Cross-modal forensics & productization

Goals

Combine signals across modalities where possible.

Provide exportable forensic reports.

Prepare for paid tiers.

Tasks

 For videos, surface separate verdicts for audio vs visuals.

 For images with text (OCR), run both image and text detectors.

 For PDFs, analyze embedded images + text.

 Generate PDF/JSON reports summarizing findings.

 Add rate limits, API keys, and usage tracking for external customers.

Exit criteria

Multimodal forensics feels cohesive in the UI.

There is a clear story for a “Pro” / enterprise tier.

Documentation exists for API consumers.


---

## 2️⃣ API spec + JSON examples

You can drop this into a `api_multimodal.md` or integrate into your OpenAPI docs.

```markdown
# CopyCat Multimodal Forensics – API

Base path: `/api/v1`

---

## Common Response Schema

Every scan endpoint returns this shape (with modality-specific `signals`):

```json
{
  "job_id": "b4d109b4-756c-4e73-8c78-6bf0f3e7c7c2",
  "modality": "image",
  "overall_verdict": "likely_ai_generated",
  "confidence": 0.87,
  "signals": {},
  "explanations": [
    "Short bullet 1...",
    "Short bullet 2..."
  ],
  "created_at": "2025-01-15T03:21:45Z"
}


overall_verdict (initial proposal):

likely_human

unclear

likely_ai_assisted

likely_ai_generated

likely_ai_modified (for images / video)

error (in case of processing failures)

confidence is 0.0–1.0, interpreted as the system’s internal calibration, not a guarantee.

POST /api/v1/scan/text

Analyze raw text (existing behavior, formalized).

Request

Content-Type: application/json

{
  "text": "Paste the essay or article here...",
  "optional_tag": "midterm_essay",
  "mode": "balanced"
}

Response (example)
{
  "job_id": "2f4b47cd-3f7e-4a9b-9bde-5f5a589ccbb0",
  "modality": "text",
  "overall_verdict": "likely_human",
  "confidence": 0.79,
  "signals": {
    "stylometry": {
      "human_similarity": 0.82,
      "ai_similarity": 0.21
    },
    "llm_likelihood": {
      "ai_prob": 0.24,
      "human_prob": 0.76
    },
    "short_text_cap": false,
    "non_english_cap": false
  },
  "explanations": [
    "Stylometric features closely match typical human writing.",
    "LLM-likelihood scores do not show strong AI probability.",
    "Sample length is sufficient for a stable estimate."
  ],
  "created_at": "2025-01-15T03:10:12Z"
}

POST /api/v1/scan/image

Analyze an image for AI generation / editing.

Request

Content-Type: multipart/form-data

Form fields:

file – image file (JPEG/PNG/WebP)

tag – optional string label

notes – optional user notes

Example (curl):

curl -X POST https://yourdomain.com/api/v1/scan/image \
  -F "file=@/path/to/image.jpg" \
  -F "tag=midjourney_check"

Response (example)
{
  "job_id": "b4d109b4-756c-4e73-8c78-6bf0f3e7c7c2",
  "modality": "image",
  "overall_verdict": "likely_ai_generated",
  "confidence": 0.91,
  "signals": {
    "metadata": {
      "has_exif": false,
      "camera_mismatch": true,
      "software": "unknown"
    },
    "compression_profile": {
      "quantization_matches_known_camera": false,
      "anomaly_score": 0.83
    },
    "ai_fingerprint": {
      "model_score": 0.94,
      "possible_family": "diffusion"
    },
    "editing": {
      "ela_suspected": true,
      "ela_hotspots": 3
    }
  },
  "explanations": [
    "No valid camera EXIF metadata was found.",
    "JPEG compression tables do not match typical camera profiles.",
    "The learned AI-fingerprint model is highly confident this image matches diffusion-style artifacts."
  ],
  "created_at": "2025-01-15T03:22:01Z"
}

POST /api/v1/scan/audio

Analyze an audio clip for AI-generated / cloned voice.

Request

Content-Type: multipart/form-data

file – audio file (WAV, MP3, M4A)

Response (example)
{
  "job_id": "86e2b3a9-913f-4b42-8c30-4aa8e5eab92b",
  "modality": "audio",
  "overall_verdict": "likely_ai_generated",
  "confidence": 0.84,
  "signals": {
    "spectrogram": {
      "ai_prob": 0.88
    },
    "prosody": {
      "pitch_stability_score": 0.91,
      "micro_variation_low": true
    },
    "noise_profile": {
      "environment_consistency": "synthetic",
      "noise_floor_db": -60.0
    },
    "breath_events": {
      "count": 0,
      "note": "No natural breath events detected in a 45-second sample."
    }
  },
  "explanations": [
    "Spectrogram-based classifier strongly favors AI-generated speech.",
    "Pitch and prosody are unusually stable for spontaneous speech.",
    "The background noise profile is consistent with synthetic rendering."
  ],
  "created_at": "2025-01-15T03:25:43Z"
}

POST /api/v1/scan/video

Analyze a video for deepfake / AI-editing.

Request

Content-Type: multipart/form-data

file – video file (MP4, MOV, etc.)

Response (example)
{
  "job_id": "39c79597-a1b1-42f5-b0e6-2bdf6be2e4de",
  "modality": "video",
  "overall_verdict": "likely_ai_modified",
  "confidence": 0.78,
  "signals": {
    "frame_analysis": {
      "avg_ai_prob": 0.73,
      "max_ai_prob": 0.96,
      "suspicious_segments": [
        {"start_s": 12.0, "end_s": 18.5, "ai_prob": 0.94},
        {"start_s": 47.0, "end_s": 55.0, "ai_prob": 0.91}
      ]
    },
    "face_landmarks": {
      "blink_rate_anomaly": true,
      "lip_sync_mismatch": true
    },
    "optical_flow": {
      "motion_anomaly_score": 0.69
    },
    "audio_channel": {
      "has_audio": true,
      "audio_ai_prob": 0.62
    }
  },
  "explanations": [
    "Multiple segments show frame-level artifacts consistent with AI-generated faces.",
    "Blink patterns and lip movements are inconsistent with natural human motion.",
    "Optical-flow analysis indicates unusual motion around the face region."
  ],
  "created_at": "2025-01-15T03:33:10Z"
}

GET /api/v1/scan/{job_id}

Retrieve a previously-completed scan.

Response

Returns the same schema as the original scan, plus any post-processed notes.

{
  "job_id": "b4d109b4-756c-4e73-8c78-6bf0f3e7c7c2",
  "modality": "image",
  "...": "..."
}


---

## 3️⃣ Image Authenticity Results Panel (HTML + CSS)

This is designed to feel like your current CopyCat panels: header bar, body, chips, “What this means” box.  
You can drop this into your main page near the text results section and tweak IDs/class names to match your existing style.

### HTML

```html
<!-- IMAGE AUTHENTICITY PANEL -->
<section id="imageAuthSection" class="panel-block">
  <header class="panel-header">
    <div class="panel-title">Image Authenticity (beta)</div>
    <div class="panel-subtitle">Check if an image is likely AI-generated or edited</div>
  </header>

  <div class="panel-body image-auth-grid">
    <!-- LEFT: upload + preview -->
    <div class="image-auth-left">
      <label class="image-dropzone" for="imageUploadInput">
        <input id="imageUploadInput" type="file" accept="image/*" hidden />
        <div class="dropzone-inner">
          <div class="dropzone-title">Drop an image here or click to upload</div>
          <div class="dropzone-hint">JPEG, PNG, WebP &middot; up to 10 MB</div>
        </div>
      </label>

      <div id="imagePreviewBox" class="image-preview-box hidden">
        <img id="imagePreview" alt="Uploaded preview" />
      </div>

      <button id="imageScanButton" class="btn btn-primary" disabled>
        Scan Image
      </button>
      <span id="imageScanStatus" class="tiny-status"></span>
    </div>

    <!-- RIGHT: verdict + signals -->
    <div class="image-auth-right">
      <!-- Overall verdict capsule -->
      <div id="imageVerdictRow" class="image-verdict-row hidden">
        <span id="imageVerdictBadge" class="verdict-badge">Awaiting scan</span>
        <div class="confidence-bar-wrap">
          <div class="confidence-bar-label">Confidence</div>
          <div class="confidence-bar-track">
            <div id="imageConfidenceFill" class="confidence-bar-fill" style="width: 0%;"></div>
          </div>
          <div id="imageConfidenceText" class="confidence-bar-text">–</div>
        </div>
      </div>

      <!-- Signal chips -->
      <div id="imageSignalsRow" class="image-signals-row hidden">
        <div class="signals-label">Signals considered</div>
        <div class="signals-chip-row">
          <span class="signal-chip" data-signal="metadata">Metadata</span>
          <span class="signal-chip" data-signal="compression">Compression profile</span>
          <span class="signal-chip" data-signal="ai_fingerprint">AI fingerprint</span>
          <span class="signal-chip" data-signal="editing">Editing / ELA</span>
        </div>
      </div>

      <!-- Explanation box -->
      <div id="imageExplainBox" class="explain-box hidden">
        <div class="explain-header">
          <span>What this means</span>
          <span class="help-dot" title="Short explanation of why this image was flagged or cleared.">?</span>
        </div>
        <ul id="imageExplainList" class="explain-list">
          <!-- Populated by JS with bullet points from `explanations` -->
        </ul>
      </div>

      <!-- Optional: ELA / heatmap placeholder -->
      <div id="imageElaBox" class="ela-box hidden">
        <div class="ela-header">
          Possible edited regions
          <span class="help-dot"
                title="Error Level Analysis highlights areas that recompress differently, which can indicate local edits.">?</span>
        </div>
        <div class="ela-content">
          <img id="imageElaPreview" alt="ELA heatmap" />
          <div class="ela-caption" id="imageElaCaption"></div>
        </div>
      </div>
    </div>
  </div>
</section>

CSS (add near your other panel styles)

Adjust variable names if any differ from your current theme tokens.

/* ----- Image Authenticity Panel ----- */

#imageAuthSection.panel-block {
  background: var(--panel);
  border-radius: 10px;
  border: 1px solid var(--border);
  padding: 16px 18px;
  margin-top: 18px;
}

#imageAuthSection .panel-header {
  margin-bottom: 12px;
}

#imageAuthSection .panel-title {
  font-size: 1rem;
  font-weight: 600;
  color: var(--text);
}

#imageAuthSection .panel-subtitle {
  font-size: 0.8rem;
  color: var(--muted);
}

.image-auth-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.1fr) minmax(0, 1.4fr);
  gap: 18px;
}

@media (max-width: 900px) {
  .image-auth-grid {
    grid-template-columns: 1fr;
  }
}

/* LEFT SIDE */

.image-dropzone {
  display: block;
  border-radius: 10px;
  border: 1px dashed var(--border);
  background: var(--panel-2);
  cursor: pointer;
}

.image-dropzone:hover {
  border-style: solid;
}

.dropzone-inner {
  padding: 16px;
  text-align: center;
}

.dropzone-title {
  font-size: 0.9rem;
  font-weight: 500;
  color: var(--text);
}

.dropzone-hint {
  font-size: 0.75rem;
  color: var(--muted);
  margin-top: 4px;
}

.image-preview-box {
  margin-top: 10px;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: var(--panel-2);
  padding: 8px;
  max-height: 220px;
  overflow: hidden;
}

.image-preview-box img {
  display: block;
  max-width: 100%;
  max-height: 200px;
  object-fit: contain;
}

#imageScanButton {
  margin-top: 10px;
}

.tiny-status {
  font-size: 0.75rem;
  margin-left: 8px;
  color: var(--muted);
}

/* RIGHT SIDE */

.image-verdict-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}

.verdict-badge {
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 0.8rem;
  font-weight: 600;
  background: var(--panel-2);
  border: 1px solid var(--border);
}

/* You can add theme-specific colors for verdicts by adding classes:
   .verdict-likely-human, .verdict-likely-ai-generated, etc. */

.confidence-bar-wrap {
  flex: 1;
  min-width: 180px;
}

.confidence-bar-label {
  font-size: 0.75rem;
  color: var(--muted);
  margin-bottom: 2px;
}

.confidence-bar-track {
  width: 100%;
  height: 6px;
  border-radius: 999px;
  background: var(--panel-2);
  overflow: hidden;
  position: relative;
}

.confidence-bar-fill {
  height: 100%;
  width: 0%;
  transition: width 0.4s ease;
  border-radius: inherit;
  background: linear-gradient(90deg, #f0b35b, #d46a6a); /* tweak to match brand */
}

.confidence-bar-text {
  font-size: 0.75rem;
  margin-top: 2px;
  color: var(--muted);
}

/* Signals */

.image-signals-row {
  margin-bottom: 12px;
}

.signals-label {
  font-size: 0.75rem;
  color: var(--muted);
  margin-bottom: 4px;
}

.signals-chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.signal-chip {
  font-size: 0.75rem;
  padding: 4px 8px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: var(--panel-2);
  cursor: default;
}

/* Explanation box */

.explain-box {
  border-radius: 8px;
  border: 1px solid var(--border);
  background: var(--panel-2);
  padding: 10px 12px;
  font-size: 0.8rem;
}

.explain-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-weight: 500;
  margin-bottom: 6px;
}

.help-dot {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 16px;
  height: 16px;
  border-radius: 999px;
  border: 1px solid var(--border);
  font-size: 0.7rem;
  cursor: default;
}

.explain-list {
  margin: 0;
  padding-left: 18px;
}

.explain-list li + li {
  margin-top: 2px;
}

/* ELA box */

.ela-box {
  margin-top: 10px;
  border-radius: 8px;
  border: 1px dashed var(--border);
  padding: 8px 10px;
  font-size: 0.8rem;
}

.ela-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 4px;
}

.ela-content img {
  display: block;
  max-width: 100%;
  border-radius: 6px;
  margin-bottom: 4px;
}

.ela-caption {
  font-size: 0.75rem;
  color: var(--muted);
}

/* Utility */

.hidden {
  display: none !important;
}

Minimal JS wiring (just the key bits)

You’ll obviously adapt this to your existing fetch logic, but here’s a skeleton:

const imageUploadInput = document.getElementById('imageUploadInput');
const imagePreviewBox = document.getElementById('imagePreviewBox');
const imagePreview = document.getElementById('imagePreview');
const imageScanButton = document.getElementById('imageScanButton');
const imageScanStatus = document.getElementById('imageScanStatus');

const imageVerdictRow = document.getElementById('imageVerdictRow');
const imageVerdictBadge = document.getElementById('imageVerdictBadge');
const imageConfidenceFill = document.getElementById('imageConfidenceFill');
const imageConfidenceText = document.getElementById('imageConfidenceText');
const imageSignalsRow = document.getElementById('imageSignalsRow');
const imageExplainBox = document.getElementById('imageExplainBox');
const imageExplainList = document.getElementById('imageExplainList');
const imageElaBox = document.getElementById('imageElaBox');
const imageElaPreview = document.getElementById('imageElaPreview');
const imageElaCaption = document.getElementById('imageElaCaption');

let imageFile = null;

imageUploadInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;

  imageFile = file;
  const url = URL.createObjectURL(file);
  imagePreview.src = url;
  imagePreviewBox.classList.remove('hidden');
  imageScanButton.disabled = false;
  imageScanStatus.textContent = '';
});

imageScanButton.addEventListener('click', async () => {
  if (!imageFile) return;

  imageScanButton.disabled = true;
  imageScanStatus.textContent = 'Scanning image…';

  const formData = new FormData();
  formData.append('file', imageFile);

  try {
    const resp = await fetch('/api/v1/scan/image', {
      method: 'POST',
      body: formData
    });
    if (!resp.ok) throw new Error('Scan failed');
    const data = await resp.json();
    renderImageScanResult(data);
  } catch (err) {
    console.error(err);
    imageScanStatus.textContent = 'Error scanning image.';
  } finally {
    imageScanButton.disabled = false;
  }
});

function renderImageScanResult(result) {
  const verdict = result.overall_verdict || 'unknown';
  const conf = Math.round((result.confidence || 0) * 100);

  imageVerdictRow.classList.remove('hidden');
  imageSignalsRow.classList.remove('hidden');
  imageExplainBox.classList.remove('hidden');

  imageVerdictBadge.textContent = verdict.replace(/_/g, ' ');
  imageConfidenceFill.style.width = `${conf}%`;
  imageConfidenceText.textContent = `${conf}%`;

  // Explanations
  imageExplainList.innerHTML = '';
  (result.explanations || []).forEach((line) => {
    const li = document.createElement('li');
    li.textContent = line;
    imageExplainList.appendChild(li);
  });

  // Optional ELA
  const editing = result.signals && result.signals.editing;
  if (editing && editing.ela_suspected && editing.ela_preview_url) {
    imageElaPreview.src = editing.ela_preview_url;
    imageElaCaption.textContent = `ELA detected ${editing.ela_hotspots || 0} possible edited region(s).`;
    imageElaBox.classList.remove('hidden');
  } else {
    imageElaBox.classList.add('hidden');
  }

  imageScanStatus.textContent = 'Scan complete.';
}