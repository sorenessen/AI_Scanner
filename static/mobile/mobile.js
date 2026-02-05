const API_BASE = "http://localhost:8080";

const scanBtn = document.getElementById("scanBtn");
const textInput = document.getElementById("textInput");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");

scanBtn.onclick = async () => {
  const text = textInput.value.trim();
  if (!text) return alert("Paste some text first.");

  statusEl.classList.remove("hidden");
  statusEl.textContent = "Scanning…";
  resultsEl.classList.add("hidden");
  resultsEl.innerHTML = "";

  try {
    const res = await fetch(`/scan`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    const ct = res.headers.get("content-type") || "";
    let payload;

    if (ct.includes("application/json")) {
      payload = await res.json();
    } else {
      // could be HTML, plain text, or a file path
      payload = { raw: await res.text(), contentType: ct };
    }

    if (!res.ok) {
      statusEl.textContent = `Scan failed (${res.status}).`;
      console.error("Scan failed:", payload);
      resultsEl.classList.remove("hidden");
      resultsEl.innerHTML = `
        <div class="card">
          <strong>Error</strong>
          <pre style="white-space: pre-wrap; margin:10px 0 0;">${escapeHtml(JSON.stringify(payload, null, 2))}</pre>
        </div>
      `;
      return;
    }

    console.log("Scan payload:", payload);
    renderResults(payload);

  } catch (e) {
    console.error("Scan exception:", e);
    statusEl.textContent = "Scan error (see console).";
  }
};

  const fileInput = document.getElementById("fileInput");
  const uploadBtn = document.getElementById("uploadBtn");
  const uploadStatus = document.getElementById("uploadStatus");

  uploadBtn.onclick = async () => {
    const file = fileInput.files && fileInput.files[0];
    if (!file) return alert("Choose a .txt, .pdf, or .docx file first.");

    uploadStatus.textContent = "Uploading…";

    try {
      const fd = new FormData();
      fd.append("file", file);
      // keep the same flow controls you already use
      fd.append("mode", "Balanced"); // or read from a selector later
      fd.append("tag", "");

      const res = await fetch(`/scan/file`, {
        method: "POST",
        body: fd,
      });

      const ct = res.headers.get("content-type") || "";
      const payload = ct.includes("application/json") ? await res.json() : { raw: await res.text() };

      if (!res.ok) {
        uploadStatus.textContent = `Upload failed (${res.status}).`;
        console.error(payload);
        return;
      }

      uploadStatus.textContent = `Scanned: ${file.name}`;
      renderResults(payload);
    } catch (e) {
      console.error(e);
      uploadStatus.textContent = "Upload error (see console).";
    }
  };

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  }[c]));
}

function renderResults(data) {
  statusEl.classList.add("hidden");
  resultsEl.classList.remove("hidden");

  if (data.source_file) {
    resultsEl.innerHTML =
      `<div class="card">
        <strong>Source</strong><br/>
        ${escapeHtml(data.source_file)} · ${escapeHtml(data.source_ext || "")}
        ${data.source_chars ? ` · ${data.source_chars} chars` : ""}
      </div>` + resultsEl.innerHTML;
  }


  const explanation = data.explanation || data.summary || "";
  const aiProb = (typeof data.calibrated_prob === "number") ? data.calibrated_prob : null;

  // Decision heuristic:
  // If calibrated_prob is present, use it. Otherwise fall back to explanation keywords.
  let decision = "N/A";
  if (aiProb !== null) {
    if (aiProb >= 0.70) decision = "Likely AI-generated";
    else if (aiProb <= 0.30) decision = "Likely human-written";
    else decision = "Mixed / uncertain";
  } else if (typeof explanation === "string") {
    const s = explanation.toLowerCase();
    if (s.includes("likely human")) decision = "Likely human-written";
    else if (s.includes("likely ai") || s.includes("ai-generated")) decision = "Likely AI-generated";
    else decision = "Result available (see summary)";
  }

  const scoreText =
    aiProb === null ? "N/A" : `${Math.round(aiProb * 100)}% AI likelihood`;

  const pplText = (typeof data.ppl === "number") ? data.ppl.toFixed(2) : "N/A";
  const burstText = (typeof data.burstiness === "number") ? data.burstiness.toFixed(2) : "N/A";
  const cat = data.category || "N/A";
  const catConf =
    (typeof data.category_conf === "number") ? `${Math.round(data.category_conf * 100)}%` : "N/A";

  // Top-10 / Top-100 % from bins (approx)
  let top10Pct = "N/A";
  let top100Pct = "N/A";
  if (data.bins && typeof data.total === "number" && data.total > 0) {
    const b10 = Number(data.bins["10"] ?? data.bins[10] ?? 0);
    const b100 = Number(data.bins["100"] ?? data.bins[100] ?? 0);
    top10Pct = `${Math.round((b10 / data.total) * 100)}%`;
    top100Pct = `${Math.round((b100 / data.total) * 100)}%`;
  }

  resultsEl.innerHTML = `
    <div class="card">
      <strong>Decision</strong><br/>
      ${escapeHtml(decision)}
    </div>

    <div class="card">
      <strong>Score</strong><br/>
      ${escapeHtml(scoreText)}
    </div>

    <div class="card">
      <strong>Summary</strong><br/>
      <span style="opacity:.92">${escapeHtml(explanation)}</span>
    </div>

    <div class="card">
      <strong>Key Metrics</strong><br/>
      <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px; margin-top:10px;">
        <div><div style="opacity:.7;font-size:12px;">PPL</div><div style="font-size:16px;">${escapeHtml(pplText)}</div></div>
        <div><div style="opacity:.7;font-size:12px;">Burstiness</div><div style="font-size:16px;">${escapeHtml(burstText)}</div></div>
        <div><div style="opacity:.7;font-size:12px;">Top-10</div><div style="font-size:16px;">${escapeHtml(top10Pct)}</div></div>
        <div><div style="opacity:.7;font-size:12px;">Top-100</div><div style="font-size:16px;">${escapeHtml(top100Pct)}</div></div>
      </div>
    </div>

    <div class="card">
      <strong>Detected Category</strong><br/>
      ${escapeHtml(cat)} <span style="opacity:.75">(${escapeHtml(catConf)})</span>
    </div>

    <details class="card">
      <summary style="cursor:pointer;"><strong>Raw JSON</strong></summary>
      <pre style="white-space: pre-wrap; margin: 10px 0 0;">${escapeHtml(JSON.stringify(data, null, 2))}</pre>
    </details>
  `;

  if (data.report_url) {
    resultsEl.innerHTML += `
      <div class="card">
        <a href="${data.report_url}" target="_blank"
           style="display:block;text-align:center;
                  padding:12px;border-radius:8px;
                  background:#4da3ff;color:black;
                  font-weight:600;text-decoration:none;">
          Open PDF Report
        </a>
      </div>
    `;
  }

  const reportUrl =
    data.report_url ||
    (data.report_file ? `/reports/${data.report_file}` : null);

  if (reportUrl) {
    resultsEl.innerHTML += `... use reportUrl ...`;
  }

  if (reportUrl) {
    resultsEl.innerHTML += `
      <div class="card" style="display:grid; gap:10px;">
        <a href="${reportUrl}" target="_blank" rel="noopener" style="...">Open PDF Report</a>
        <button id="copyLinkBtn" style="...">Copy report link</button>
      </div>
    `;
    document.getElementById("copyLinkBtn").onclick = async () => {
      await navigator.clipboard.writeText(location.origin + reportUrl);
      alert("Copied!");
    };
  }

  document.getElementById("clearBtn").onclick = () => {
    textInput.value = "";
    if (fileInput) fileInput.value = "";
    resultsEl.classList.add("hidden");
    resultsEl.innerHTML = "";
    statusEl.classList.add("hidden");
    if (uploadStatus) uploadStatus.textContent = "";
  };
}


// function renderResults(data) {
//   statusEl.classList.add("hidden");
//   resultsEl.classList.remove("hidden");

//   resultsEl.innerHTML = `
//     <div class="card">
//       <strong>Decision</strong><br/>
//       ${data.decision ?? "N/A"}
//     </div>

//     <div class="card">
//       <strong>Score</strong><br/>
//       ${data.score ?? "N/A"}
//     </div>

//     <div class="card">
//       <strong>Summary</strong><br/>
//       ${data.summary ?? ""}
//     </div>
//   `;
// }
