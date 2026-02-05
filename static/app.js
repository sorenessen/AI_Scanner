// --- Demo Mode Sample Texts ---
const DEMO_SAMPLES = {
  "news": `Breaking: A new AI-driven policing system has sparked debate...`,
  "story": `Once the lantern flickered out, Mara realized she wasn't alone...`,
  "essay": `The socioeconomic impact of automation continues to reshape...`,
  "code": `def analyze_text(t):\n    return {"len": len(t)}`,
};

// --- Demo presets ---
const DEMO_PRESET_LABELS = {
  random: "(random curated sample)",
  news:   "News / op-ed",
  story:  "Story / narrative",
  essay:  "Student-style essay",
  code:   "Short code snippet",
};

let currentDemoPreset = "random";

function initDemoPresetSelect() {
  const sel = document.getElementById("demoPreset");
  if (!sel) return;

  // Build options from the label map
  sel.innerHTML = Object.entries(DEMO_PRESET_LABELS)
    .map(([value, label]) => `<option value="${value}">${label}</option>`)
    .join("");

  sel.value = currentDemoPreset;

  sel.addEventListener("change", (e) => {
    const v = String(e.target.value || "").trim();
    currentDemoPreset = DEMO_PRESET_LABELS[v] ? v : "random";
  });
}


/* -------- v0.3.5 -------- */
window.__pdCentroids = 0;
/* ---------- helpers ---------- */
const LCARS_HISTORY_KEY = "copycat_scan_history_v1";

const $ = s => document.querySelector(s);
const api = (p,opts={}) => fetch(p,opts).then(r=>{ if(!r.ok) throw new Error(r.status+' '+r.statusText); return r.json(); });
const setMsg = (id,txt)=>{ const el=$( '#'+id ); if(el) el.textContent = txt; };
function setFinalizeEnabled(on){ $("#btnFinalize").disabled = !on; }
function setExportEnabled(on){
  const c=$("#btnExport"), d=$("#btnDownload");
  if(c) c.disabled = !on;
  if(d) d.disabled = !on;
}

/* Drift Diagnostics */
async function driftAnalyze(text){
  const r = await fetch("/drift/analyze", {
    method:"POST", headers:{ "content-type":"application/json" },
    body: JSON.stringify({ text })
  });
  if(!r.ok) throw new Error(await r.text());
  return r.json();
}

/* Drift Diagnostics */
async function driftAnalyze(text) {
  const r = await fetch("/drift/analyze", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ text })
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

/* Drift compare: scan vs live sample */
async function driftCompare(scanText, liveText) {
  const resp = await fetch("/drift/compare", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      scan_text: scanText,   // <-- must match the Python model field names
      live_text: liveText
    })
  });

  if (!resp.ok) {
    throw new Error(await resp.text());
  }
  return resp.json();
}

// Example render (stick this where you show scan extras)
async function showDriftForCurrent(){
  const text = document.querySelector("#text")?.value || "";
  if(!text.trim()){ alert("Paste text first."); return; }
  const j = await driftAnalyze(text);
  const el = document.getElementById("liveResult") || document.body;
  el.innerHTML =
    `<div class="kv" style="margin-top:8px">
       <div>Paragraphs</div><div class="mono">${j.paragraphs}</div>
       <div>Avg adj. sim</div><div class="mono">${j.avg_adjacent_sim}</div>
       <div>Std adj. sim</div><div class="mono">${j.std_adjacent_sim}</div>
       <div>Risk</div><div class="mono">${j.risk}</div>
       <div>Score (human-like)</div><div class="mono">${j.score}</div>
     </div>`;
}


/* Toggle help on click (outside click / Esc closes) */
document.addEventListener("click", (e)=>{
  const dot = e.target.closest(".help-dot");
  document.querySelectorAll(".help-dot[data-open='1']").forEach(el=>{ if(el!==dot) el.removeAttribute("data-open"); });
  if (dot){ dot.setAttribute("data-open", dot.getAttribute("data-open")==="1" ? "0" : "1"); }
});
document.addEventListener("keydown",(e)=>{
  if(e.key==="Escape"){ document.querySelectorAll(".help-dot[data-open='1']").forEach(el=>el.removeAttribute("data-open")); closeLive(); }
  if((e.metaKey||e.ctrlKey) && e.key==="Enter"){ finalizeCompute(); }
});

/* UI scale */
$("#uiScale").addEventListener("input", (e)=>{
  const v = Math.max(120, Math.min(220, parseInt(e.target.value||"150",10)));
  document.documentElement.style.setProperty("--scale", (v/100).toString());
});

/* Drawer */
const drawer = $("#liveDrawer");
const openLive = ()=>{ drawer.classList.add("open"); drawer.setAttribute("aria-hidden","false"); };
const closeLive = ()=>{ drawer.classList.remove("open"); drawer.setAttribute("aria-hidden","true"); };
$("#btnLive").addEventListener("click", openLive);
$("#btnCloseLive").addEventListener("click", closeLive);

/* ---- Live drawer resize ---- */
(function initLiveResize(){
  const handle = $("#liveResizeHandle");
  if (!handle) return;

  let dragging = false;

  // clamp + apply height (in px)
  function setLiveHeight(px){
    const min = 160;
    const max = Math.max(260, window.innerHeight * 0.85);
    const clamped = Math.min(max, Math.max(min, px));
    document.documentElement.style.setProperty("--live-h", clamped + "px");
  }

  function onMove(e){
    if (!dragging) return;
    const clientY = (e.touches && e.touches[0]) ? e.touches[0].clientY : e.clientY;
    const h = window.innerHeight - clientY;   // distance from cursor to bottom
    setLiveHeight(h);
  }

  function stopDrag(){
    if (!dragging) return;
    dragging = false;
    document.removeEventListener("mousemove", onMove);
    document.removeEventListener("mouseup", stopDrag);
    document.removeEventListener("touchmove", onMove);
    document.removeEventListener("touchend", stopDrag);
    document.removeEventListener("touchcancel", stopDrag);
  }

  function startDrag(e){
    dragging = true;
    e.preventDefault();
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", stopDrag);
    document.addEventListener("touchmove", onMove, { passive:false });
    document.addEventListener("touchend", stopDrag);
    document.addEventListener("touchcancel", stopDrag);
  }

  handle.addEventListener("mousedown", startDrag);
  handle.addEventListener("touchstart", startDrag, { passive:false });

  // keep height sensible when the window resizes
  window.addEventListener("resize", ()=>{
    const h = parseFloat(getComputedStyle(document.documentElement)
      .getPropertyValue("--live-h")) || (window.innerHeight * 0.4);
    setLiveHeight(h);
  });
})();


/* UI writers */
function setProbFill(p){
  const pct   = Math.round((p || 0) * 100);
  const label = pct + "%";

  // Text in the Result row
  const pctEl = $("#probPct");
  if (pctEl) pctEl.textContent = label;

  // Text in the center of the donut
  const centerEl = $("#probPctCenter");
  if (centerEl) centerEl.textContent = label;

  // Donut stroke
  const donut = document.querySelector(".prob-donut .donut-val");
  if (donut){
    const clamped = Math.max(0, Math.min(1, p || 0));
    const radius = 16;                           // matches SVG r
    const circumference = 2 * Math.PI * radius;  // ~100.53

    donut.style.strokeDasharray  = `${circumference} ${circumference}`;
    donut.style.strokeDashoffset = `${circumference * (1 - clamped)}`;
  }



  // Fallback: if bar still exists anywhere, keep it in sync
  const f = $("#probFill");
  if (f){
    f.style.width = pct + "%";
    f.classList.remove("good","warn","bad");
    f.classList.add(
      p >= 0.65 ? "bad" :
      p >= 0.35 ? "warn" :
                  "good"
    );
  }
}

function setVerdict(v){
  const pill = $("#verdictPill");
  pill.textContent = v || "—";
  const L = (v||"").toLowerCase();
  const c = L.includes("human") ? "#163626" :
            L.includes("inconclusive") ? "#2f2a12" : "#3a1b1b";
  pill.style.background = c; pill.style.borderColor = "var(--border)";
}
function setModeBadge(m){ $("#activeModeBadge").textContent = "Mode: " + (m||"—"); }
function shortExplain(r){
  if (r?.explanation) return r.explanation;
  const top10 = r?.bins ? Math.round((r.bins[10]/Math.max(1,r.total))*100) : 0;
  return `${r?.verdict||""} — PPL≈${(r?.ppl||0).toFixed(1)}, Top10≈${top10}%`.trim();
}

/* numeric helpers */
const pickNum = (...c)=>{ for(const v of c){ const n=Number(v); if(Number.isFinite(n)) return n; } return null; };
const setTxt = (sel,val,d=3)=>{ const el=$(sel); if(!el) return; if(val==null){ el.textContent="—"; return; } const n=Number(val); el.textContent=Number.isFinite(n)?n.toFixed(d):String(val); };

/* Token table */
function cleanTok(t){
  if (t === "Ċ") return "↵";
  if (t === "Ġ") return "␠";
  return t.replace(/^Ġ/," ")
          .replace(/âĢĿ/g,"“").replace(/âĢĺ/g,"”")
          .replace(/âĢĶ/g,"‘").replace(/âĢĺ/g,"’");
}

function safeSetHTML(el, html){ if(el) el.innerHTML = html; }
function safeSetText(el, txt){ if(el) el.textContent = txt; }

function fillTokenTable(tokens){
  const tbWrap = document.querySelector("#tokTable tbody");
  if (!tbWrap) return;                 // v0.3.4 fallback (no table or id changed)
  tbWrap.innerHTML = "";
  (tokens||[]).forEach((t,i)=>{
    const tr = document.createElement("tr");
    tr.innerHTML =
      `<td class="mono">${i+1}</td>`+
      `<td class="mono">${cleanTok(t.t)}</td>`+
      `<td class="mono">${t.rank}</td>`+
      `<td class="mono">${(t.p||0).toFixed(6)}</td>`;
    tbWrap.appendChild(tr);
  });
}

function setBandBadge(band, pctText){
  const badge = document.getElementById('bandBadge');
  if (!badge) return;
  const dot  = badge.querySelector('.badge-dot');
  const text = document.getElementById('bandText');
  const pct  = document.getElementById('bandPct');

  badge.classList.remove('band-human','band-mixed','band-ai');

  let cls='band-mixed', dotCls='dot-mixed', label = band || 'Inconclusive';
  const b = (band||'').toLowerCase();
  if (b.includes('human')) { cls='band-human'; dotCls='dot-human'; }
  else if (b.includes('ai')) { cls='band-ai'; dotCls='dot-ai'; }

  badge.classList.add(cls);
  if (dot)  dot.className = 'badge-dot ' + dotCls;
  if (text) text.textContent = label;
  if (pct)  pct.textContent  = pctText ? `• ${pctText}` : '';
  badge.style.display = 'inline-flex';
}


function renderExplain(explain, fullJson) {
  const wrap     = document.getElementById('explainWrap');
  const head     = document.getElementById('explainHeadline');
  const why      = document.getElementById('explainWhy');
  const fix      = document.getElementById('explainFix');
  const notes    = document.getElementById('explainNotes');
  const mini     = document.getElementById('teacherMini');
  const title    = document.getElementById('explainTitle');
  const panel    = document.getElementById('explainPanel');
  const openBtn  = document.getElementById('btnExplainPanel');

  if (!wrap) return;

  // No explanation payload → disable button, keep panel closed
  if (!explain) {
    if (openBtn) {
      openBtn.disabled = true;
      openBtn.classList.add('disabled');
    }
    if (panel) {
      panel.classList.remove('open');
      panel.setAttribute('aria-hidden', 'true');
    }
    return;
  }

  // Enable “What this means” button once we have data
  if (openBtn) {
    openBtn.disabled = false;
    openBtn.classList.remove('disabled');
  }

  // Badge (if present)
  if (document.getElementById('bandBadge')) {
    setBandBadge(explain.band, explain.ai_likelihood_pct);
  }

  // Header + sections
  if (title) {
    // keep text in case you ever surface it somewhere else
    title.textContent = 'What this means';
  }

  safeSetHTML(head, `<b>${explain.headline || ''}</b>`);

  const asList = a => Array.isArray(a) ? a.filter(Boolean) : [];

  const whyList = asList(explain.why).map(s => `<li>${s}</li>`).join('');
  safeSetHTML(
  why,
  whyList
    ? `<b>Why CopyCat thinks this</b><ul class="explain-list">${whyList}</ul>`
    : ''
);

  const fixList = asList(explain.what_to_fix).map(s => `<li>${s}</li>`).join('');
  safeSetHTML(
  fix,
  fixList
    ? `<b>How to strengthen your draft</b><ul class="explain-list">${fixList}</ul>`
    : ''
);

  const notesText = asList(explain.notes).join(' ');
  safeSetHTML(
  notes,
  notesText ? `<b>Extra context</b> ${notesText}` : ''
);

  // Compact teacher row
  if (mini) {
    const t = explain.teacher_report || {};
    const parts = [];
    if (t.nearest_style) parts.push(`Closest style: ${t.nearest_style}`);
    if (t.drift_score != null) parts.push(`Drift score: ${Number(t.drift_score).toFixed(3)}`);
    if (t.pd_overlap_j != null) parts.push(`PD overlap J: ${t.pd_overlap_j}`);
    mini.textContent = parts.join(' • ');
  }

  // PD note
  if (notes) {
    if (!Number(window.__pdCentroids || 0)) {
      const msg = "Public-domain fingerprinting is off on this server (no centroids loaded). PD overlap shows ‘—’. Add JSON centroids to ./pd_fingerprints/ and restart.";
      const prefix = notes.innerHTML ? "<br/>" : "";
      notes.innerHTML += prefix + `<b>PD note:</b> ${msg}`;
    }
  }

  // No internal hide/show toggle anymore—panel handles visibility.
}


function renderFingerprint(result) {
  const box      = document.getElementById("fpBox");
  const labelEl  = document.getElementById("fpLabel");
  const bandEl   = document.getElementById("fpBand");
  const topEl    = document.getElementById("fpTop");
  const bandCard = document.getElementById("fpBandCard");
  const spreadEl = document.getElementById("fpSpread");
  const notesEl  = document.getElementById("fpNotes");

  if (!box) return;

  // Try a few common shapes for the payload
  const fp = result?.llm_fingerprint
          || result?.fingerprint
          || result?.fp
          || null;

  // Nothing to show → hide the card and clear fields
  if (!fp) {
    box.style.display = "none";
    if (labelEl)  labelEl.textContent  = "—";
    if (bandEl)   bandEl.textContent   = "—";
    if (topEl)    topEl.textContent    = "—";
    if (bandCard) bandCard.textContent = "—";
    if (spreadEl) spreadEl.textContent = "—";
    if (notesEl)  notesEl.textContent  = "—";
    return;
  }

  // Expanded fallbacks so we handle both old/new backend shapes
  const top = fp.nearest
           ?? fp.top_style
           ?? fp.top
           ?? fp.family
           ?? fp.label
           ?? "—";

  const band = fp.band
            ?? fp.band_label
            ?? fp.bucket
            ?? "—";

  const spread = fp.spread
              ?? fp.drift
              ?? fp.pd_spread
              ?? fp.pd
              ?? null;

  const notes = fp.notes
             || fp.note
             || fp.comment
             || fp.explain
             || "";

  // Summary row pills
  if (labelEl)  labelEl.textContent  = top;
  if (bandEl)   bandEl.textContent   = band;

  // Card contents
  if (topEl)    topEl.textContent    = top;
  if (bandCard) bandCard.textContent = band;
  if (spreadEl) spreadEl.textContent = spread == null ? "—" : String(spread);
  if (notesEl)  notesEl.textContent  = notes || "—";

  box.style.display = "block";
}

function renderDrift(raw) {
  const box = document.getElementById("driftBox");
  const content = document.getElementById("driftContent");
  if (!box || !content) return;

  const safeNum = (v, d = 3) => {
  if (typeof v !== "number" || !Number.isFinite(v)) return "—";
  return v.toFixed(d);
  };


  // No payload → hide card
  if (!raw) {
    box.style.display = "none";
    content.innerHTML = "";
    return;
  }

  // New: backend nests the drift info under `semantic_drift`
  const j = raw.semantic_drift || raw;

  // Handle the “not available” case if backend ever sends it
  if (j.available === false) {
    box.style.display = "block";
    content.innerHTML = `
      <div>Paragraphs</div><div class="mono">${j.paragraphs ?? "—"}</div>
      <div>Status</div><div class="mono">
        ${j.reason === "single_paragraph"
          ? "Need at least two paragraphs to measure semantic drift."
          : (j.reason || "Drift diagnostics not available.")}
      </div>
    `;
    return;
  }

  // Normal case: we have real stats
  box.style.display = "block";
  content.innerHTML = `
    <div>Paragraphs</div><div class="mono">${j.paragraphs ?? raw.paragraphs ?? "—"}</div>
    <div>Avg adjacent similarity</div><div class="mono">${safeNum(j.avg_adjacent_sim)}</div>
    <div>Std adjacent similarity</div><div class="mono">${safeNum(j.std_adjacent_sim)}</div>
    <div>Risk</div><div class="mono">${safeNum(j.risk)}</div>
    <div>Score (human-like)</div><div class="mono">${safeNum(j.score)}</div>
  `;
}

function renderLiveDrift(raw) {
  const box = document.getElementById("liveDriftBox");
  const content = document.getElementById("liveDriftContent");
  if (!box || !content) return;

  const safeNum = (v, d = 3) =>
    (typeof v === "number" && Number.isFinite(v)) ? v.toFixed(d) : "—";

  // No payload at all → generic message
  if (!raw) {
    box.style.display = "block";
    content.innerHTML =
      `<div class="mono">Drift compare not available for this sample.</div>`;
    return;
  }

  // Many backends will wrap the actual compare result
  const cmp = raw.compare || raw.c || raw;

  // Only trust explicit fields from the backend
  const sim = (typeof cmp.similarity === "number")
    ? cmp.similarity
    : (typeof cmp.sim === "number" ? cmp.sim : null);

  const score = (typeof cmp.score === "number") ? cmp.score : null;
  const risk  = (typeof cmp.risk  === "number") ? cmp.risk  : null;

  const overlap = (typeof cmp.overlap === "number")
    ? cmp.overlap
    : (typeof cmp.jaccard === "number" ? cmp.jaccard : null);

  const haveAny = [sim, score, risk, overlap].some(
    v => typeof v === "number" && Number.isFinite(v)
  );

  // If the backend doesn’t provide actual compare metrics yet,
  // don’t fabricate numbers – just explain that it’s not wired.
    if (!haveAny) {
    box.style.display = "block";
    content.innerHTML = `
      <div class="mono">
        Drift compare is not fully wired on this build:
        the backend isn’t sending numeric similarity / risk metrics yet.
        <br><br>
        TODO (next release): implement /drift/compare in the backend
        and return similarity / overlap / risk fields so this panel
        can show real numbers.
      </div>
    `;
    return;
  }


  // Normal case: show what the backend really sent
  box.style.display = "block";
  content.innerHTML = `
    <div>Similarity</div><div class="mono">${safeNum(sim)}</div>
    <div>Score (human-like)</div><div class="mono">${safeNum(score)}</div>
    <div>Risk</div><div class="mono">${safeNum(risk)}</div>
    <div>Overlap</div><div class="mono">${safeNum(overlap)}</div>
  `;
}



function initLiveTabs(){
  const tabs = document.querySelectorAll(".live-tab");
  if (!tabs.length) return;

  const liveResult = document.getElementById("liveResult");
  const driftBox   = document.getElementById("liveDriftBox");

  function activate(name){
    tabs.forEach(t => {
      const isActive = (t.dataset.tab === name);
      t.classList.toggle("active", isActive);
      t.setAttribute("aria-selected", isActive ? "true" : "false");
    });

    if (liveResult) liveResult.style.display = (name === "summary") ? "block" : "none";
    if (driftBox)   driftBox.style.display   = (name === "drift")    ? "block" : "none";
  }

  tabs.forEach(t => {
    t.addEventListener("click", () => {
      const tabName = t.dataset.tab || "summary";
      activate(tabName);
    });
  });

  // default view
  activate("summary");
}

function renderLlmFingerprint(result) {
  const box      = document.getElementById("llmBox");
  const bandPill = document.getElementById("llmBand");
  const famEl    = document.getElementById("llmFamily");
  const confEl   = document.getElementById("llmConf");
  const humanEl  = document.getElementById("llmHuman");
  const legendEl = document.getElementById("llmLegend");
  const canvas   = document.getElementById("llmCanvas");

  if (!box) return;

  const fp =
    result?.llm_fingerprint ||
    result?.fingerprint ||
    result?.fp ||
    null;

  if (!fp || fp.available === false) {
    box.style.display = "none";
    if (bandPill) bandPill.textContent = "—";
    if (famEl)    famEl.textContent    = "—";
    if (confEl)   confEl.textContent   = "—";
    if (humanEl)  humanEl.textContent  = "—";
    if (legendEl) legendEl.textContent = "";
    if (canvas && canvas.getContext) {
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    return;
  }

  box.style.display = "block";

  const family =
    fp.nearest_family ||
    fp.family ||
    fp.nearest ||
    fp.top_style ||
    fp.top ||
    fp.label ||
    "Unknown";

  const dist = fp.distribution || fp.similarity || {};
  const labels = Object.keys(dist);
  const values = labels.map(k => Number(dist[k]) || 0);

  const conf = typeof fp.confidence === "number" ? fp.confidence : null;
  const humanScore =
    typeof fp.human_score === "number"
      ? fp.human_score
      : (typeof fp.human === "number" ? fp.human : null);

  // Band pill based on humanScore
  let band = "—";
  if (typeof humanScore === "number") {
    if (humanScore >= 0.8) band = "Human-like";
    else if (humanScore <= 0.2) band = "Model-like";
    else band = "Mixed";
  }

  if (famEl)    famEl.textContent    = family;
  if (bandPill) bandPill.textContent = band;
  if (confEl)   confEl.textContent   = conf == null ? "—" : `${(conf * 100).toFixed(1)}%`;
  if (humanEl)  humanEl.textContent  = humanScore == null ? "—" : humanScore.toFixed(3);

  // Legend line
  if (legendEl) {
    const bits = [];
    if (labels.length) {
      bits.push(
        "Distribution: " +
        labels.map(k => `${k} ${(dist[k] * 100).toFixed(1)}%`).join(" · ")
      );
    }
    legendEl.textContent = bits.join("  |  ");
  }

  // Simple bar visualization for human_score, if we have it
  if (canvas && canvas.getContext && typeof humanScore === "number") {
    const ctx = canvas.getContext("2d");
    const w   = canvas.width;
    const h   = canvas.height;

    ctx.clearRect(0, 0, w, h);

    ctx.fillStyle = "rgba(255,255,255,0.08)";
    ctx.fillRect(0, h / 2 - 2, w, 4);

    const barW = Math.max(4, Math.min(w, w * Math.max(0, Math.min(1, humanScore))));
    ctx.fillStyle = "rgba(79,134,255,0.9)";
    ctx.fillRect(0, h / 2 - 6, barW, 12);
  }
}



/* ---------- Live verification ---------- */
let lastScanMetrics = null;
let lastScanResult  = null;
let lastExportText  = "";
let didScan = false;
let lastScanText = "";

function tokenizeWords(text){ return (text.toLowerCase().match(/[a-z’']+|\d+|[^\s\w]/g) || []); }
const FN_WORDS = new Set(("a,an,the,of,to,in,for,with,on,at,by,from,as,that,this,which,who,whom,whose,and,or,but,if,then,so,because,while,where,about,into,over,after,before,between,through,during,without,within,against,under,above,across,around,per,via,is,am,are,was,were,be,been,being").split(","));
function punctEntropy(text){
  const punct = text.match(/[.,;:!?()-]/g) || []; if(!punct.length) return 0;
  const counts={}; punct.forEach(p=>counts[p]=(counts[p]||0)+1);
  const N=punct.length; let H=0; for(const k in counts){ const p=counts[k]/N; H-=p*Math.log2(p); } return H;
}
function simpleMetrics(text){
  const tokens = tokenizeWords(text);
  const words = tokens.filter(t=>/^[a-z’']+$/.test(t));
  const n = words.length || 1;
  const uniq = new Set(words).size;
  const fnCount = words.reduce((a,w)=>a+(FN_WORDS.has(w)?1:0),0);
  const func_ratio = fnCount/n;
  const hapax_ratio = uniq/n;
  const sentences = text.split(/(?<=[.!?])\s+/).filter(s=>s.trim().length>0);
  const lens = sentences.map(s => (s.match(/\b\w+\b/g)||[]).length);
  const sent_mean = lens.length ? lens.reduce((a,b)=>a+b,0)/lens.length : 0;
  const sent_var  = lens.length ? lens.reduce((a,b)=>a+(b-sent_mean)**2,0)/lens.length : 0;
  const pent = punctEntropy(text);
  return { func_ratio, hapax_ratio, sent_mean, sent_var, punct_entropy: pent };
}
function cosineSimilarity(a,b){
  const keys=["func_ratio","hapax_ratio","sent_mean","sent_var","punct_entropy"];
  let dot=0,na=0,nb=0; for(const k of keys){ const x=Number(a[k])||0, y=Number(b[k])||0; dot+=x*y; na+=x*x; nb+=y*y; }
  return (na&&nb)?(dot/(Math.sqrt(na)*Math.sqrt(nb))):0;
}
function renderLiveCompare(userM, refM){
  const sim = Math.max(0,Math.min(1,cosineSimilarity(userM,refM)));
  const pct = Math.round(sim*100);
  const fields=[["Function word ratio","func_ratio",3],["Hapax ratio","hapax_ratio",3],["Sent. mean","sent_mean",2],["Sent. var","sent_var",2],["Punct entropy","punct_entropy",3]];
  let html = `<div class="row" style="gap:12px"><strong>Match (quick):</strong> <span class="pill">${pct}%</span></div>`;
  html += `<div class="kv" style="margin-top:8px">`;
  for(const [label,key,d] of fields){
    const u=Number(userM[key])||0, r=Number(refM[key])||0, diff=u-r;
    html += `<div>${label}</div><div class="mono">you ${u.toFixed(d)} • ref ${r.toFixed(d)} • Δ ${(diff>=0?"+":"")}${diff.toFixed(d)}</div>`;
  }
  html += `</div>`;
  return { html, pct };
}

function initAdvToggleButton() {
  const details = document.getElementById("advMetricsBox");
  const btn     = document.getElementById("advToggleBtn");
  if (!details || !btn) return;

  function sync() {
    btn.textContent = details.open
      ? "Hide advanced metrics"
      : "Show advanced metrics";
  }

  details.addEventListener("toggle", sync);
  btn.addEventListener("click", () => {
    details.open = !details.open;
    sync();
  });

  sync();
}


/* ============================
 * LCARS SIDE PANEL: fill tabs after a scan
 * ============================ */

function updateLcarsPanel(result, driftRaw) {
  if (!result) return;

  const prob = Number(result.calibrated_prob ?? result.prob ?? 0);
  const probPct = Math.round(prob * 100);
  const verdict = result.verdict || "—";
  const mode    = result.mode || "Balanced";
  const category = result.category || "—";
  const model   = result.model_name || "—";
  const tag     = (result.tag || result.input_tag || "").trim() || "—";
  const pdJ     = (typeof result.pd_overlap_j === "number")
    ? result.pd_overlap_j.toFixed(3)
    : "—";

  // Normalize drift payload shape
  const drift = driftRaw && (driftRaw.semantic_drift || driftRaw);
  const driftParas = drift ? (drift.paragraphs ?? "—") : "—";
  const driftRisk  = drift && typeof drift.risk === "number"
    ? drift.risk.toFixed(3)
    : "—";
  const driftScore = drift && typeof drift.score === "number"
    ? drift.score.toFixed(3)
    : "—";

  /* ----------------- Transparency tab ------------------- */
  const tabTransparency = document.getElementById("tab-transparency");
  if (tabTransparency) {
    tabTransparency.innerHTML = `
      <h3 class="lcars-section-title">Overall verdict</h3>
      <div class="lcars-kv">
        <div>Verdict</div>
        <div class="mono">
          <span class="lcars-pill-inline ${
            verdict.toLowerCase().includes("human") ? "good" :
            verdict.toLowerCase().includes("inconclusive") ? "warn" : "bad"
          }">
            ${verdict}
          </span>
        </div>

        <div>Calibrated probability</div>
        <div class="mono">${probPct}%</div>

        <div>Category</div>
        <div class="mono">${category}</div>

        <div>Mode</div>
        <div class="mono">${mode}</div>

        <div>Tag</div>
        <div class="mono">${tag}</div>

        <div>Model</div>
        <div class="mono">${model}</div>

        <div>PD overlap J</div>
        <div class="mono">${pdJ}</div>
      </div>
      <p class="lcars-subtext">
        CopyCat exposes the same signals our engine uses internally &mdash; the subtle
        fingerprints that shape a text&rsquo;s behavior more than its meaning. We surface
        pacing rhythm, token variety, repetition curves, and model-style drift so you
        can see what a quick skim or gradebook can&rsquo;t. None of these signals alone
        decide anything; they only form a behavioral profile. Think of this tab as
        the forensic readout: the raw style telemetry that informs the overall
        assessment &mdash; openly, honestly, and without any hidden scoring.
      </p>
    `;
  }

  /* ----------------- Writing Tutor tab ------------------- */
  const tabTutor = document.getElementById("tab-tutor");
  if (tabTutor) {
    const explain = result.explain || {};
    const whyList  = Array.isArray(explain.why) ? explain.why.filter(Boolean) : [];
    const fixList  = Array.isArray(explain.what_to_fix) ? explain.what_to_fix.filter(Boolean) : [];
    const notes    = Array.isArray(explain.notes) ? explain.notes.join(" ") : (explain.notes || "");

    const whyHtml = whyList.length
      ? `<b>Why we think this:</b><ul class="explain-list">${whyList.map(s => `<li>${s}</li>`).join("")}</ul>`
      : `<span class="lcars-empty">No detailed rationale available for this scan.</span>`;

    const fixHtml = fixList.length
      ? `<b>How to strengthen human signal:</b><ul class="explain-list">${fixList.map(s => `<li>${s}</li>`).join("")}</ul>`
      : `<span class="lcars-empty">No specific suggestions returned for this scan.</span>`;

    const notesHtml = notes
      ? `<b>Notes:</b> ${notes}`
      : `<span class="lcars-empty">No additional notes.</span>`;

    tabTutor.innerHTML = `
      <h3 class="lcars-section-title">Human writing tutor</h3>
      <div class="explain-section">${whyHtml}</div>
      <div class="explain-section">${fixHtml}</div>
      <div class="explain-section explain-note">${notesHtml}</div>
      <p class="lcars-subtext">
        This panel offers gentle, real-world polish rooted in <em>your</em> sample&rsquo;s
        style. These suggestions never try to &ldquo;beat detectors&rdquo; or disguise your work;
        they strengthen clarity, pacing, sentence variation, and your personal voice.
        CopyCat treats you like an author, not a suspect &mdash; we highlight ways to
        reinforce your authentic style, reduce accidental monotony, and sharpen
        narrative flow, especially when you&rsquo;re writing fast under a deadline.
      </p>
        <ul class="lcars-history-list">
  <li>Stabilize your tone from intro to conclusion.</li>
  <li>Blend short and long sentences so the rhythm feels lived-in.</li>
  <li>Use concrete nouns and verbs that sound like you, not a template.</li>
</ul>

    `;
  }

  /* ----------------- Style Match tab --------------------- */
  const tabStyle = document.getElementById("tab-style");
  if (tabStyle) {
    const S = result.stylometry
      || result.style
      || result.metrics?.stylometry
      || result.stats?.stylometry
      || {};

    const safe = (v, d = 3) =>
      (typeof v === "number" && Number.isFinite(v)) ? v.toFixed(d) : "—";

    const funcRatio  = safe(S.func_ratio ?? S.function_word_ratio);
    const hapaxRatio = safe(S.hapax_ratio ?? S.hapax);
    const sentMean   = safe(S.sent_mean ?? S.sentence_mean, 2);
    const sentVar    = safe(S.sent_var ?? S.sentence_var, 2);
    const pent       = safe(S.punct_entropy ?? S.punctuation_entropy);

    tabStyle.innerHTML = `
      <h3 class="lcars-section-title">Style match (scan)</h3>
      <div class="lcars-kv">
        <div>Function-word ratio</div><div class="mono">${funcRatio}</div>
        <div>Hapax ratio</div><div class="mono">${hapaxRatio}</div>
        <div>Sentence mean length</div><div class="mono">${sentMean}</div>
        <div>Sentence variance</div><div class="mono">${sentVar}</div>
        <div>Punctuation entropy</div><div class="mono">${pent}</div>
      </div>
      <p class="lcars-subtext">
        Style Match shows how your stylistic fingerprint compares with common
        human-writing baselines and with known model families. A high match score
        doesn&rsquo;t automatically mean &ldquo;AI-written,&rdquo; and a low score doesn&rsquo;t grant
        a free pass. It simply reflects shared characteristics: rhythm, token
        preferences, repetition patterns, and structural habits. Use this tab to
        understand which parts of your style stand out as uniquely yours, and which
        resemble well-known stylistic families on both the human and model side.
      </p>

    `;
  }

  /* ----------------- Tips tab ---------------------------- */
  const tabTips = document.getElementById("tab-tips");
  if (tabTips) {
    const explain = result.explain || {};
    const fixList  = Array.isArray(explain.what_to_fix) ? explain.what_to_fix.filter(Boolean) : [];

    const bullets = fixList.length
      ? fixList
      : [
          "Add a few personal details or concrete examples to strengthen human signal.",
          "Vary sentence length and rhythm to avoid overly uniform structure.",
          "Rephrase any obviously template-like phrases into your natural voice."
        ];

    tabTips.innerHTML = `
      <h3 class="lcars-section-title">Tips to sound more like you</h3>
      <ul class="explain-list">
        ${bullets.map(s => `<li>${s}</li>`).join("")}
      </ul>
      <p class="lcars-subtext">
        Tips are small, low-friction adjustments that keep your writing anchored to
        a strong personal signature, especially in longer assignments or multi-part
        projects. CopyCat focuses on writer empowerment, not trickery: we nudge you
        toward clearer pacing, varied sentence openings, and a steady narrative tone
        so your work reads more like <em>you</em>, not like a generic template or a
        flattened model voice. Treat this tab as practical coaching, not a
        checklist to &ldquo;pass&rdquo; a detector.
      </p>

    `;
  }

  /* ----------------- Safety tab -------------------------- */
  const tabSafety = document.getElementById("tab-safety");
  if (tabSafety) {
    tabSafety.innerHTML = `
      <h3 class="lcars-section-title">Safety diagnostics (advisory)</h3>
      <div class="lcars-kv">
        <div>Toxicity</div><div class="mono">— (not yet implemented)</div>
        <div>Sentiment</div><div class="mono">— (not yet implemented)</div>
        <div>Bias flags</div><div class="mono">— (not yet implemented)</div>
        <div>Drift risk</div><div class="mono">${driftRisk}</div>
        <div>Drift score (human-like)</div><div class="mono">${driftScore}</div>
        <div>Paragraphs analyzed</div><div class="mono">${driftParas}</div>
      </div>
      <p class="lcars-subtext muted">
        Safety diagnostics highlight potential issues but do not replace
        human review or institutional policy.
      </p>
      <ul class="lcars-history-list">
        <li>Keep your thesis and closing paragraph in the same emotional register.</li>
        <li>Vary how you start sentences to avoid mechanical repetition.</li>
        <li>Reserve complex wording for ideas that actually need it.</li>
      </ul>

    `;
  }

  /* ----------------- Mission Log + History --------------- */
  const now = new Date();
  const tsShort = now.toLocaleString(undefined, {
    year: "numeric", month: "short", day: "2-digit",
    hour: "2-digit", minute: "2-digit"
  });

  // Persist history in localStorage
  let history = [];
  try {
    history = JSON.parse(localStorage.getItem(LCARS_HISTORY_KEY) || "[]");
    if (!Array.isArray(history)) history = [];
  } catch {
    history = [];
  }

  history.unshift({
    ts: tsShort,
    verdict,
    probPct,
    mode,
    tag
  });

  history = history.slice(0, 20);
  try {
    localStorage.setItem(LCARS_HISTORY_KEY, JSON.stringify(history));
  } catch (e) {
    console.warn("Failed to store LCARS history:", e);
  }

  const tabHistory = document.getElementById("tab-history");
  if (tabHistory) {
    if (!history.length) {
      tabHistory.innerHTML = `
        <h3 class="lcars-section-title">Recent scans</h3>
        <p class="lcars-empty muted">No previous scans recorded.</p>
      `;
    } else {
      tabHistory.innerHTML = `
        <h3 class="lcars-section-title">Recent scans</h3>
        <ul class="lcars-history-list">
          ${history.map(h => `
            <li>
              <span class="mono">${h.ts}</span> —
              <strong>${h.verdict}</strong>
              (<span class="mono">${h.probPct}%</span>)
              • Mode: <span class="mono">${h.mode}</span>
              • Tag: <span class="mono">${h.tag}</span>
            </li>
          `).join("")}
        </ul>
      `;
    }
  }

  const tabMission = document.getElementById("tab-mission");
  if (tabMission) {
    tabMission.innerHTML = `
      <h3 class="lcars-section-title">Mission log entry</h3>
      <div class="lcars-kv">
        <div>Stardate</div><div class="mono">${tsShort}</div>
        <div>Verdict</div><div class="mono">${verdict}</div>
        <div>Calibrated prob.</div><div class="mono">${probPct}%</div>
        <div>Runtime mode</div><div class="mono">${mode}</div>
      <p class="lcars-subtext">
        Mission Log is your stardate-style record of this scan. It captures detector
        status, runtime mode, stylistic fingerprint IDs, and any notable anomalies
        in one compact entry. In LCARS terms, this is your console report: a trace
        of how CopyCat saw the text at scan time, so you can correlate verdicts,
        tuning changes, and revisions without guesswork or hidden state.
      </p>

    `;
  }
}



/* ---------- Scan ---------- */
async function runScan(){
  const demoOn = $("#demoToggle").checked;

  // Whatever is in the textarea right now
  let text = $("#text").value.trim();
  const hadUserText = text.length > 0;

  // If demo is on AND user hasn’t typed anything,
  // pull from the selected preset instead of random.
  if (demoOn && !hadUserText) {
    const presetSel  = $("#demoPreset");
    const presetKey  = presetSel ? presetSel.value : "random";

    if (presetKey === "random") {
      // Old behavior: random curated sample
      const keys = Object.keys(DEMO_SAMPLES);
      const pick = keys[Math.floor(Math.random() * keys.length)];
      text = DEMO_SAMPLES[pick];
    } else if (DEMO_SAMPLES[presetKey]) {
      // Use the requested preset
      text = DEMO_SAMPLES[presetKey];
    } else {
      // Safety fallback: still pick something valid
      const keys = Object.keys(DEMO_SAMPLES);
      const pick = keys[Math.floor(Math.random() * keys.length)];
      text = DEMO_SAMPLES[pick];
    }

    // Reflect the chosen text in the textarea so it’s obvious
    $("#text").value = text;
  }

  // If *still* no text, bail with error message
  if (!text) {
    $("#err").style.display = "block";
    $("#err").textContent   = "Please enter some text.";
    return;
  }

  lastScanText = text;
  $("#err").style.display = "none";
  $("#btnScan").disabled  = true;

  // Status text: distinguish demo vs normal
  $("#scanStatus").textContent =
    (demoOn && !hadUserText) ? "Scanning demo sample…" : "Scanning…";

  try {
    const body = {
      text,
      tag:  $("#tag").value.trim() || null,
      mode: $("#mode").value || null,
      demo_mode: demoOn || false   // nice to have for backend / logs
    };

    const r = await api("/scan", {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify(body)
});

didScan        = true;
lastScanResult = r;


renderExplain(r.explain, r);
renderFingerprint(r);

// Drift diagnostics
let driftForPanel = null;
try {
  const drift = await driftAnalyze(text);
  driftForPanel = drift;        // <— this is the one we pass to LCARS
  renderDrift(drift);
} catch (e) {
  console.warn("Drift diagnostics failed:", e);
  renderDrift(null);
}

// LLM fingerprint card
try {
  renderLlmFingerprint(r);
} catch (e) {
  console.warn("LLM fingerprint render failed:", e);
  renderLlmFingerprint(null);
}

// NEW: feed data into LCARS side panel tabs
try {
  updateLcarsPanel(r, driftForPanel);
} catch (e) {
  console.warn("LCARS panel update failed:", e);
}


    const rawBox = document.getElementById("rawDump");
    if (rawBox) rawBox.textContent = JSON.stringify(r, null, 2);
    document.getElementById("rawJsonBox")?.removeAttribute("open");

    $("#result").style.display = "block";

    // Keep your existing bar logic (even though the bar is visually hidden)
    setProbFill(r.calibrated_prob || 0);
    setVerdict(r.verdict || "—");

    // NEW: drive the donut + center label from calibrated_prob
    const prob = (typeof r.calibrated_prob === "number") ? r.calibrated_prob : null;

    const pctCenter = $("#probPctCenter");
    if (pctCenter) {
      pctCenter.textContent = (prob !== null)
        ? `${Math.round(prob * 100)}% AI`
        : "—";
    }

    updateProbDonut(prob);

    const NS = r.nonsense_signals || {};
    const top10Pct  = Number.isFinite(NS.top10)
      ? Math.round(NS.top10*100)
      : (r.bins ? Math.round((r.bins[10]  / Math.max(1,r.total))*100) : null);
    const top100Pct = Number.isFinite(NS.top100)
      ? Math.round(NS.top100*100)
      : (r.bins ? Math.round((r.bins[100] / Math.max(1,r.total))*100) : null);

    $("#top10").textContent   = top10Pct  ?? "—";
    $("#top100").textContent  = top100Pct ?? "—";
    $("#ppl").textContent     = (NS.ppl ?? r.ppl ?? 0).toFixed(2);
    $("#burst").textContent   = (NS.burst ?? r.burstiness ?? 0).toFixed(3);
    $("#category").textContent = r.category
      ? `${r.category} (${Math.round((r.category_conf||0)*100)}%)`
      : "—";
    $("#modelName").textContent = r.model_name || "—";
    $("#modeEcho").textContent  = r.mode || "—";
    $("#pdj").textContent       = (r.pd_overlap_j ?? 0).toFixed(3);
    $("#tagEcho").textContent   = body.tag || "—";
    $("#explainShort").textContent = shortExplain(r);


    fillTokenTable(r.per_token || []);

    const S = r.stylometry || r.style || r.metrics?.stylometry || r.stats?.stylometry || {};
    const N2 = r.nonsense_signals || r.nonsense || r.metrics?.nonsense || r.stats?.nonsense || {};

    setTxt("#sty_func",  pickNum(S.func_ratio, S.function_word_ratio, S.funcWordsRatio), 3);
    setTxt("#sty_hapax", pickNum(S.hapax_ratio, S.hapax, S.hapaxRatio), 3);
    setTxt("#sty_smean", pickNum(S.sent_mean, S.sentence_mean, S.sentMean), 2);
    setTxt("#sty_svar",  pickNum(S.sent_var, S.sentence_var, S.sentVar), 2);
    setTxt("#sty_mlps",  pickNum(S.mlps, S.delta_logp_mean, S.dlogp_mean, S.deltaLogpMean), 4);
    setTxt("#sty_mlpsv", pickNum(S.mlps_var, S.delta_logp_var, S.dlogp_var, S.deltaLogpVar), 4);
    setTxt("#sty_pent",  pickNum(S.punct_entropy, S.punctuation_entropy, S.punctEntropy), 3);

    setTxt("#ns_rhyme", pickNum(N2.rhyme_density, N2.rhyme, N2.rhymeDensity), 3);
    setTxt("#ns_meter", pickNum(N2.meter_cv, N2.meter, N2.meterCV), 3);
    setTxt("#ns_inv",   pickNum(N2.invented_ratio, N2.invented, N2.inventedRatio), 3);
    setTxt("#ns_lex",   pickNum(N2.lex_hits, N2.lexHits), 0);
    setTxt("#ns_sem",   pickNum(N2.semantic_disc, N2.semantic_discontinuity, N2.semanticDisc), 3);

    lastScanMetrics = {
      func_ratio:    Number(S.func_ratio ?? S.function_word_ratio ?? 0),
      hapax_ratio:   Number(S.hapax_ratio ?? S.hapax ?? 0),
      sent_mean:     Number(S.sent_mean ?? S.sentence_mean ?? 0),
      sent_var:      Number(S.sent_var  ?? S.sentence_var  ?? 0),
      punct_entropy: Number(S.punct_entropy ?? S.punctuation_entropy ?? 0),
    };

    $("#btnStartLive").disabled = false;
    $("#liveInput").disabled    = false;
    setFinalizeEnabled(true);
    setExportEnabled(false);
    setMsg("exportStatus","");
  } catch (e) {
    $("#err").style.display = "block";
    $("#err").textContent   = "Scan failed: " + e.message;
  } finally {
    $("#btnScan").disabled = false;
    $("#scanStatus").textContent = "";
  }
}

function updateProbDonut(prob) {
  const donut = document.querySelector("#result .donut-val");
  if (!donut || typeof prob !== "number") {
    return; // safely no-op if missing or no probability
  }

  // Must match r="16" from your SVG <circle> in the donut
  const radius = 16;
  const circumference = 2 * Math.PI * radius;

  donut.style.strokeDasharray  = `${circumference} ${circumference}`;
  donut.style.strokeDashoffset = String(circumference * (1 - prob));
}


/* Demos */
async function loadDemos(){
  $("#btnDemo").disabled = true;
  try {
    const preset = currentDemoPreset;
    let text = "";

    if (preset === "random") {
      const keys = Object.keys(DEMO_SAMPLES);
      if (!keys.length) {
        $("#text").value = "No demo samples configured.";
        return;
      }
      const pick = keys[Math.floor(Math.random() * keys.length)];
      text = DEMO_SAMPLES[pick];
    } else {
      text = DEMO_SAMPLES[preset] || "";
      if (!text) {
        $("#text").value = `Demo preset "${preset}" not available.`;
        return;
      }
    }

    $("#text").value = text;
  } catch (e) {
    $("#text").value = "Failed to load demo sample: " + e.message;
  } finally {
    $("#btnDemo").disabled = false;
  }
}


/* Live typing */
let liveStartTs = null, liveTimerId = null, liveComputingTimer = null;

async function finalizeCompute(){
  const typed = $("#liveInput").value.trim();

  if (!didScan){
    setMsg("liveResult","Run a scan first to have a reference.");
    setExportEnabled(false);
    renderLiveDrift(null);
    return;
  }

  if (!typed) {
    setMsg("liveResult","No sample typed.");
    setExportEnabled(false);
    renderLiveDrift(null);
    return;
  }

  const wordCount = tokenizeWords(typed).filter(t=>/^[a-z’']+$/i.test(t)).length;
  if (wordCount < 60) {
    setMsg("liveResult","Type at least ~60 words for a stable compare.");
    setExportEnabled(false);
    renderLiveDrift(null);
    return;
  }

  const userM = simpleMetrics(typed);
  const { html, pct } = renderLiveCompare(userM, lastScanMetrics);
  $("#liveResult").innerHTML = html;

  lastExportText = buildExportText(lastScanResult, userM, lastScanMetrics, pct);
  setExportEnabled(true);
  setMsg("exportStatus","Summary ready.");

  // NEW: call /drift/compare for scan vs live sample
  try {
    if (lastScanText) {
      const drift = await driftCompare(lastScanText, typed);
      renderLiveDrift(drift);
    } else {
      renderLiveDrift(null);
    }
  } catch (e) {
    console.warn("Live drift compare failed:", e);
    renderLiveDrift({ note: "Drift compare unavailable: " + e.message });
  }
}

function tickTimer(){
  if(!liveStartTs) return;
  const s = Math.max(0, Math.floor((Date.now()-liveStartTs)/1000));
  $("#liveTimer").textContent = s + "s";
  if (s >= 90) { $("#liveRough").textContent = "Computing…"; stopLive(true); }
}

function updateLiveHUD() {
  const liveInputEl = $("#liveInput");
  if (!liveInputEl) return;

  const typed  = liveInputEl.value;
  const tokens = tokenizeWords(typed);
  const words  = tokens.filter(t => /^[a-z’']+$/i.test(t));
  const count  = words.length;

  const tokensEl = $("#liveTokens");
  if (tokensEl) {
    tokensEl.textContent = String(count);
  }

  const pct = Math.min(100, Math.round((count / 120) * 100));

  // Narrow progress bar (inside the header strip)
  const bar = $("#liveProgress");
  if (bar) {
    bar.style.width = pct + "%";
  }

  // LEN: xx% label
  const lenEl = $("#liveLen");
  if (lenEl) {
    lenEl.textContent = pct + "%";
  }

  // Wide progress bar under the compare section, if present
  const wide = document.getElementById("liveProgressWide");
  if (wide) {
    wide.style.width = pct + "%";
  }

  if (lastScanMetrics && count >= 20) {
    const sim = Math.round(
      100 *
        Math.max(
          0,
          Math.min(1, cosineSimilarity(simpleMetrics(typed), lastScanMetrics))
        )
    );
    const roughEl = $("#liveRough");
    if (roughEl) {
      roughEl.textContent = `~${sim}%`;
    }
  } else {
    const roughEl = $("#liveRough");
    if (roughEl) {
      roughEl.textContent = "—";
    }
  }
}

function startLive() {
  $("#liveResult").textContent = "";
  $("#btnStartLive").disabled = true;
  setFinalizeEnabled(true);

  // Safely reset the progress bar
  const bar = $("#liveProgress");
  if (bar) {
    bar.style.width = "0%";
  }

  // Safely reset the “LEN: 0%” label
  const lenEl = $("#liveLen");
  if (lenEl) {
    lenEl.textContent = "0%";
  }

  $("#liveRough").textContent = "—";

  liveStartTs = Date.now();

  // Kick the HUD every 500ms (timer + word count + quick match)
  liveTimerId = setInterval(() => {
    tickTimer();
    updateLiveHUD();
  }, 500);
}


function stopLive(autoFinalize=false){
  clearInterval(liveTimerId); liveTimerId = null; liveStartTs = null;
  $("#btnStartLive").disabled = false;
  setFinalizeEnabled(true);

  // NEW: always turn off fun zone when live stops
  if (window.setFunZoneActive) window.setFunZoneActive(false);

  if (autoFinalize){
    clearTimeout(liveComputingTimer);
    liveComputingTimer = setTimeout(() => finalizeCompute(), 600);
  }
}


/* ----- Export (plain text summary) ----- */
function buildExportText(scan, userM, refM, simPct){
  const ts = new Date().toISOString();
  const sampleWords = tokenizeWords($("#liveInput").value).filter(t=>/^[a-z’']+$/i.test(t)).length;
  const runtime = $("#activeModeBadge")?.textContent.replace("Mode: ","") || (scan?.mode ?? "—");
  const ensemOn = (($("#ensemBadge")?.textContent)||"").toLowerCase().includes("on") ? "on" : "off";
  const pdn = Number(window.__pdCentroids || 0);

  const lines = [];
  if (scan){
    const prob = Number(scan.calibrated_prob ?? 0);
    const probPct = Math.round(prob*100);
    lines.push("CopyCat — Scan Summary");
    lines.push(`Timestamp: ${ts}`);
    lines.push(`Verdict: ${scan.verdict ?? "—"} (${probPct}%)`);
    if (scan.category) lines.push(`Category: ${scan.category} (${Math.round((scan.category_conf||0)*100)}%)`);
    lines.push(`Model: ${scan.model_name ?? "—"} | Runtime: ${runtime} | Ensemble: ${ensemOn}`);
    if (typeof scan.pd_overlap_j === "number") lines.push(`PD overlap J: ${scan.pd_overlap_j.toFixed(3)}`);
    lines.push(`PD fingerprints loaded: ${pdn > 0 ? pdn : "none"}`);
    lines.push("");
    lines.push("Quick style match (Live Verification):");
    lines.push(`Match: ${simPct}%`);
    lines.push(`Live words: ${sampleWords}`);
  }
  const fields = [
    ["Function word ratio","func_ratio",3],
    ["Hapax ratio","hapax_ratio",3],
    ["Sent. mean","sent_mean",2],
    ["Sent. var","sent_var",2],
    ["Punct entropy","punct_entropy",3],
  ];
  lines.push("");
  for(const [label,key,d] of fields){
    const u = Number(userM?.[key] ?? 0);
    const r = Number(refM?.[key] ?? 0);
    const diff = u - r;
    lines.push(`${label}: you ${u.toFixed(d)} | ref ${r.toFixed(d)} | Δ ${(diff>=0?"+":"")}${diff.toFixed(d)}`);
  }
  return lines.join("\n");
}

async function copyExport(){
  if (!lastExportText){ setMsg("exportStatus","Nothing to copy yet."); return; }
  try{
    await navigator.clipboard.writeText(lastExportText);
    setMsg("exportStatus","Copied to clipboard.");
  }catch{
    // Fallback for Safari/iOS/permission blocks
    const ta = document.createElement("textarea");
    ta.value = lastExportText;
    ta.style.position="fixed"; ta.style.opacity="0";
    document.body.appendChild(ta);
    ta.focus(); ta.select();
    try{ document.execCommand("copy"); setMsg("exportStatus","Copied (fallback)."); }
    catch(e){ setMsg("exportStatus","Copy failed: " + e.message); }
    finally{ document.body.removeChild(ta); }
  }
}
function downloadExportTxt(){
  if (!lastExportText){ setMsg("exportStatus","Nothing to download yet."); return; }
  const blob = new Blob([lastExportText], {type: "text/plain"});
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href = url;
  a.download = `copycat_summary_${new Date().toISOString().slice(0,19).replace(/[:T]/g,'-')}.txt`;
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
  setMsg("exportStatus","Downloaded .txt.");
}

/* ----- Config I/O (unchanged API surface, but shows version bump) ----- */
async function loadVersion(){
  try{
    const v = await api("/version");
    $("#version").textContent = `v${v.version || "0.3.8"} • ${v.model} • ${v.device} ${v.dtype} • ensemble=${v.ensemble?"on":"off"}`;
    $("#ensemBadge").textContent = v.ensemble ? "Ensemble: on" : "Ensemble: off";
   const pdn = Number(v.fingerprint_centroids ?? 0);
   const pd  = $("#pdBadge");
    if (pd) {
      const has = pdn > 0;

      // Text
      pd.textContent = has ? `PD: ${pdn}` : "PD: none";

      // Remove any old inline styles so CSS can take over
      pd.removeAttribute("style");

      // Toggle dark-theme state classes
      pd.classList.remove("pd-ok", "pd-none");
      pd.classList.add(has ? "pd-ok" : "pd-none");

      // Accessibility label
      pd.setAttribute(
        "aria-label",
        has
          ? `${pdn} PD fingerprint centroids loaded`
          : "No PD fingerprint centroids loaded"
      );

      window.__pdCentroids = pdn;
    }


  }catch{
    $("#version").textContent = "v0.3.5";
  }
}
async function loadConfig(){
  try{
    const j = await api("/config"); const s = j.settings || {};
    $("#cfg_mode").value = s.mode || "Balanced";
    $("#use_ensemble").checked = !!s.use_ensemble;
    $("#cfg_min_tokens").value = s.min_tokens_strong ?? 180;
    $("#cfg_short_cap").checked = !!s.short_cap;
    $("#cfg_max_conf_short").value = (s.max_conf_short ?? 0.35);
    $("#cfg_non_en_cap").value = (s.non_en_cap ?? 0.15);
    $("#cfg_en_thresh").value = (s.en_thresh ?? 0.70);
    $("#cfg_max_unstable").value = (s.max_conf_unstable ?? 0.35);
    $("#cfg_abstain_low").value = (s.abstain_low ?? 0.35);
    $("#cfg_abstain_high").value = (s.abstain_high ?? 0.65);
    setModeBadge(s.mode || "Balanced");
    $("#mode").value = s.mode || "Balanced";
  }catch(e){ $("#saveStatus").textContent = "Failed to load settings: "+e.message; }
}
async function saveConfig(){
  const body = {
    mode: $("#cfg_mode").value,
    short_cap: $("#cfg_short_cap").checked,
    min_tokens_strong: parseInt($("#cfg_min_tokens").value || "180",10),
    use_ensemble: $("#use_ensemble").checked,
    non_en_cap: parseFloat($("#cfg_non_en_cap").value || "0.15"),
    en_thresh: parseFloat($("#cfg_en_thresh").value || "0.70"),
    max_conf_unstable: parseFloat($("#cfg_max_unstable").value || "0.35"),
    max_conf_short: parseFloat($("#cfg_max_conf_short").value || "0.35"),
    abstain_low: parseFloat($("#cfg_abstain_low")?.value || "0.35"),
    abstain_high: parseFloat($("#cfg_abstain_high")?.value || "0.65"),
  };
  try{
    $("#btnSave").disabled = true; $("#saveStatus").textContent = "Saving…";
    const j = await api("/config",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify(body)});
    $("#saveStatus").textContent = "Saved.";
    setModeBadge(j.settings.mode);
    $("#mode").value = j.settings.mode;
    await loadVersion();
  }catch(e){ $("#saveStatus").textContent = "Save failed: "+e.message; }
  finally{ $("#btnSave").disabled = false; }
}


/* ---------- File upload + scan (UI-only, minimal surface) ---------- */
async function uploadAndScanFile(file){
  if (!file) return;

  const statusEl = document.getElementById("uploadStatus");
  const setUploadStatus = (t)=>{ if(statusEl) statusEl.textContent = t || ""; };

  // Basic allowlist
  const name = (file.name || "").toLowerCase();
  const ok = name.endsWith(".txt") || name.endsWith(".pdf") || name.endsWith(".docx");
  if (!ok){
    setUploadStatus("Unsupported file type. Use .txt, .pdf, or .docx.");
    return;
  }

  const modeVal = document.getElementById("mode")?.value || "";
  const tagVal  = document.getElementById("tag")?.value?.trim() || "";

  setUploadStatus("Uploading…");
  document.getElementById("btnUpload")?.setAttribute("disabled","disabled");

  try{
    const fd = new FormData();
    fd.append("file", file);
    // optional metadata (server will ignore if absent)
    fd.append("mode", modeVal);
    fd.append("tag", tagVal);

    const resp = await fetch("/scan/file", { method:"POST", body: fd });
    if(!resp.ok) throw new Error(await resp.text());
    const r = await resp.json();

    // If backend provided extracted text, populate textarea so the user sees what's being scanned.
    const extracted = (typeof r._source_text === "string") ? r._source_text : "";
    if (extracted) {
      const ta = document.getElementById("text");
      if (ta) ta.value = extracted;
    }

    await applyScanResult(r, extracted || "", tagVal, modeVal);

    setUploadStatus("Uploaded.");
  }catch(e){
    setUploadStatus("Upload failed: " + (e?.message || String(e)));
  }finally{
    document.getElementById("btnUpload")?.removeAttribute("disabled");
    // clear input so selecting the same file twice still fires change
    const fi = document.getElementById("fileInput");
    if (fi) fi.value = "";
    window.setTimeout(()=>setUploadStatus(""), 2200);
  }
}

/* Reuse the same render pipeline as a normal scan */
async function applyScanResult(r, sourceText, tagVal, modeVal){
  didScan        = true;
  lastScanResult = r;

  // Keep a usable source text for drift compare (when available)
  lastScanText = (sourceText || "").trim();

  renderExplain(r.explain, r);
  renderFingerprint(r);

  // Drift diagnostics (best-effort)
  let driftForPanel = null;
  try {
    if (lastScanText) {
      const drift = await driftAnalyze(lastScanText);
      driftForPanel = drift;
      renderDrift(drift);
    } else {
      renderDrift(null);
    }
  } catch (e) {
    console.warn("Drift diagnostics failed:", e);
    renderDrift(null);
  }

  // LLM fingerprint card
  try { renderLlmFingerprint(r); }
  catch (e) { console.warn("LLM fingerprint render failed:", e); renderLlmFingerprint(null); }

  // LCARS side panel
  try { updateLcarsPanel(r, driftForPanel); }
  catch (e) { console.warn("LCARS panel update failed:", e); }

  const rawBox = document.getElementById("rawDump");
  if (rawBox) rawBox.textContent = JSON.stringify(r, null, 2);
  document.getElementById("rawJsonBox")?.removeAttribute("open");

  $("#result").style.display = "block";

  // Probability + verdict
  setProbFill(r.calibrated_prob || 0);
  setVerdict(r.verdict || "—");

  const prob = (typeof r.calibrated_prob === "number") ? r.calibrated_prob : null;
  const pctCenter = $("#probPctCenter");
  if (pctCenter) pctCenter.textContent = (prob !== null) ? `${Math.round(prob * 100)}% AI` : "—";
  updateProbDonut(typeof prob === "number" ? prob : 0);

  // Metrics readout
  const NS = r.nonsense_signals || {};
  const top10Pct  = Number.isFinite(NS.top10)
    ? Math.round(NS.top10*100)
    : (r.bins ? Math.round((r.bins[10]  / Math.max(1,r.total))*100) : null);
  const top100Pct = Number.isFinite(NS.top100)
    ? Math.round(NS.top100*100)
    : (r.bins ? Math.round((r.bins[100] / Math.max(1,r.total))*100) : null);

  $("#top10").textContent   = top10Pct  ?? "—";
  $("#top100").textContent  = top100Pct ?? "—";
  $("#ppl").textContent     = (NS.ppl ?? r.ppl ?? 0).toFixed(2);
  $("#burst").textContent   = (NS.burst ?? r.burstiness ?? 0).toFixed(3);
  $("#category").textContent = r.category
    ? `${r.category} (${Math.round((r.category_conf||0)*100)}%)`
    : "—";
  $("#modelName").textContent = r.model_name || "—";
  $("#modeEcho").textContent  = r.mode || modeVal || "—";
  $("#pdj").textContent       = (r.pd_overlap_j ?? 0).toFixed(3);
  $("#tagEcho").textContent   = tagVal || "—";
  $("#explainShort").textContent = shortExplain(r);

  fillTokenTable(r.per_token || []);

  const S = r.stylometry || r.style || r.metrics?.stylometry || r.stats?.stylometry || {};
  const N2 = r.nonsense_signals || r.nonsense || r.metrics?.nonsense || r.stats?.nonsense || {};

  setTxt("#sty_func",  pickNum(S.func_ratio, S.function_word_ratio, S.funcWordsRatio), 3);
  setTxt("#sty_hapax", pickNum(S.hapax_ratio, S.hapax, S.hapaxRatio), 3);
  setTxt("#sty_smean", pickNum(S.sent_mean, S.sentence_mean, S.sentMean), 2);
  setTxt("#sty_svar",  pickNum(S.sent_var, S.sentence_var, S.sentVar), 2);
  setTxt("#sty_mlps",  pickNum(S.mlps, S.delta_logp_mean, S.dlogp_mean, S.deltaLogpMean), 4);
  setTxt("#sty_mlpsv", pickNum(S.mlps_var, S.delta_logp_var, S.dlogp_var, S.deltaLogpVar), 4);
  setTxt("#sty_pent",  pickNum(S.punct_entropy, S.punctuation_entropy, S.punctEntropy), 3);

  setTxt("#ns_rhyme", pickNum(N2.rhyme_density, N2.rhyme, N2.rhymeDensity), 3);
  setTxt("#ns_meter", pickNum(N2.meter_cv, N2.meter, N2.meterCV), 3);
  setTxt("#ns_inv",   pickNum(N2.invented_ratio, N2.invented, N2.inventedRatio), 3);
  setTxt("#ns_lex",   pickNum(N2.lex_hits, N2.lexHits), 0);
  setTxt("#ns_sem",   pickNum(N2.semantic_disc, N2.semantic_discontinuity, N2.semanticDisc), 3);

  lastScanMetrics = {
    func_ratio:    Number(S.func_ratio ?? S.function_word_ratio ?? 0),
    hapax_ratio:   Number(S.hapax_ratio ?? S.hapax ?? 0),
    sent_mean:     Number(S.sent_mean ?? S.sentence_mean ?? 0),
    sent_var:      Number(S.sent_var  ?? S.sentence_var  ?? 0),
    punct_entropy: Number(S.punct_entropy ?? S.punctuation_entropy ?? 0),
  };

  $("#btnStartLive").disabled = false;
  $("#liveInput").disabled    = false;
  setFinalizeEnabled(true);
  setExportEnabled(false);
  setMsg("exportStatus","");
}

/* Wire */

function initFileUpload() {
  const fileInput   = document.getElementById("fileInput");
  const uploadBtn   = document.getElementById("uploadBtn");
  const uploadStatus= document.getElementById("uploadStatus");

  if (!fileInput || !uploadBtn) return;

  uploadBtn.addEventListener("click", async () => {
    const file = fileInput.files && fileInput.files[0];
    if (!file) return alert("Choose a .txt, .pdf, or .docx file first.");

    uploadBtn.disabled = true;
    if (uploadStatus) uploadStatus.textContent = "Uploading…";

    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("mode", document.getElementById("mode")?.value || "Balanced");
      fd.append("tag",  document.getElementById("tag")?.value?.trim() || "");

      const resp = await fetch("/scan/file", { method: "POST", body: fd });

      const ct = resp.headers.get("content-type") || "";
      const payload = ct.includes("application/json")
        ? await resp.json()
        : { raw: await resp.text(), contentType: ct };

      if (!resp.ok) {
        if (uploadStatus) uploadStatus.textContent = `Upload failed (${resp.status}).`;
        console.error("Upload failed:", payload);
        alert("Upload failed. See console.");
        return;
      }

      if (uploadStatus) uploadStatus.textContent = `Scanned: ${file.name}`;

      // Treat it like a normal scan result so everything updates
      didScan        = true;
      lastScanResult = payload;
      lastScanText   = payload.input_text || ""; // if backend returns it; otherwise leave

      // Reuse your existing rendering pipeline
      renderExplain(payload.explain, payload);
      renderFingerprint(payload);

      let driftForPanel = null;
      try {
        // If backend extracted text and returned it, use that
        const t = payload.text || payload.input_text || "";
        if (t.trim()) {
          driftForPanel = await driftAnalyze(t);
          renderDrift(driftForPanel);
        } else {
          renderDrift(null);
        }
      } catch (e) {
        console.warn("Drift diagnostics failed:", e);
        renderDrift(null);
      }

      try { renderLlmFingerprint(payload); } catch(e){ renderLlmFingerprint(null); }
      try { updateLcarsPanel(payload, driftForPanel); } catch(e){}

      // Update the main result UI like runScan does
      $("#result").style.display = "block";
      setProbFill(payload.calibrated_prob || 0);
      setVerdict(payload.verdict || "—");
      updateProbDonut(payload.calibrated_prob);

      // If your backend returns report_url/report_file, show it in the UI
      const rawBox = document.getElementById("rawDump");
      if (rawBox) rawBox.textContent = JSON.stringify(payload, null, 2);
      document.getElementById("rawJsonBox")?.removeAttribute("open");

    } catch (e) {
      console.error(e);
      if (uploadStatus) uploadStatus.textContent = "Upload error (see console).";
    } finally {
      uploadBtn.disabled = false;
    }
  });
}


$("#btnScan").addEventListener("click", runScan);
$("#btnDemo").addEventListener("click", loadDemos);
$("#btnClear").addEventListener("click", ()=>{
  $("#text").value="";
  $("#result").style.display="none";
  $("#err").style.display="none";
  $("#btnStartLive").disabled = true;
  $("#liveInput").disabled = true;
  didScan = false;
  setFinalizeEnabled(false);
  setExportEnabled(false);
  lastExportText = "";
  setMsg("exportStatus","");


  // v0.3.7 — reset drift UI
  const driftBox = document.getElementById("driftBox");
  const driftContent = document.getElementById("driftContent");
  if (driftBox) driftBox.style.display = "none";
  if (driftContent) driftContent.innerHTML = "";
    const liveDriftContent = document.getElementById("liveDriftContent");
  if (liveDriftContent) liveDriftContent.innerHTML = "";

  // reset tabs back to Summary
  initLiveTabs();


});

$("#btnSave").addEventListener("click", saveConfig);
$("#btnReset").addEventListener("click", ()=>location.reload());
$("#btnStartLive").addEventListener("click", startLive);
$("#btnFinalize").addEventListener("click", finalizeCompute);
$("#btnExport").addEventListener("click", copyExport);
$("#btnDownload").addEventListener("click", downloadExportTxt);

// Upload & scan (file picker)
const fileInputEl = document.getElementById("fileInput");
const btnUploadEl = document.getElementById("btnUpload");
if (btnUploadEl && fileInputEl) {
  btnUploadEl.addEventListener("click", () => fileInputEl.click());
  fileInputEl.addEventListener("change", () => {
    const f = fileInputEl.files && fileInputEl.files[0];
    if (f) uploadAndScanFile(f);
  });
}


// Keep HUD updated as you type in Live Verification
const liveInputEl = document.getElementById("liveInput");
if (liveInputEl) {
  liveInputEl.addEventListener("input", updateLiveHUD);
}


(async function init(){
  await loadVersion();
  await loadConfig();
  initDemoPresetSelect();
  initLiveTabs();
  setFinalizeEnabled(false);
  setExportEnabled(false);
  initSmartTooltips();
  initAdvToggleButton();
  initFileUpload();

})();

(function () {
  const CAT_FRAME_W = 256;
  const CAT_FRAME_H = 256;
  const CAT_COLS    = 4;
  const CAT_TOTAL   = 16;

  let catFrame = 0;
  let catTimer = null;

  function setCatFrame(el, frameIndex) {
    const col = frameIndex % CAT_COLS;
    const row = Math.floor(frameIndex / CAT_COLS);
    const x = -col * CAT_FRAME_W;
    const y = -row * CAT_FRAME_H;
    el.style.backgroundPosition = `${x}px ${y}px`;
  }

  function startCatSprite() {
    const cat = document.querySelector('#liveFunZone .cat');
    if (!cat) return;
    if (catTimer) return;

    catFrame = 0;
    setCatFrame(cat, catFrame);

    catTimer = window.setInterval(() => {
      catFrame = (catFrame + 1) % CAT_TOTAL;
      setCatFrame(cat, catFrame);
    }, 1000 / 12);
  }

  function stopCatSprite() {
    if (catTimer) {
      window.clearInterval(catTimer);
      catTimer = null;
    }
  }

  function setFunZoneActive(on) {
    const zone = document.getElementById('liveFunZone');
    if (!zone) return;

    if (on) {
      zone.classList.add('active');
      startCatSprite();
    } else {
      zone.classList.remove('active');
      stopCatSprite();
    }
  }

  // Expose to the rest of the app (startLive / stopLive use this)
  window.setFunZoneActive = setFunZoneActive;
})();

  

function initSmartTooltips() {
  const dots = document.querySelectorAll(".help-dot");

  function adjustPlacement(dot) {
    if (!dot) return;

    // Reset any previous override
    dot.classList.remove("tip-l");

    const tip = dot.querySelector(".tip");
    if (!tip) return;

    const rect = dot.getBoundingClientRect();
    const vw = window.innerWidth || document.documentElement.clientWidth;

    const tipWidth = tip.offsetWidth || 260;  // fallback

    const rightSpace = vw - rect.right;

    // If not enough room on the right, flip this one to the left
    if (rightSpace < tipWidth + 16) {
      dot.classList.add("tip-l");
    }
  }

  dots.forEach(dot => {
    dot.addEventListener("mouseenter", () => adjustPlacement(dot));
    dot.addEventListener("focus",      () => adjustPlacement(dot));
  });

  window.addEventListener("resize", () => {
    dots.forEach(adjustPlacement);
  });
}


(function initTheme() {
  const sel = document.getElementById("themeSelect");
  if (!sel) return;

  const THEME_KEY = "copycat_theme";
  const allowed = new Set(["dark-professional", "light-paper", "lcars"]);

  // Read saved theme or default to dark-professional
  const stored = localStorage.getItem(THEME_KEY);
  const initial = allowed.has(stored) ? stored : "dark-professional";

  document.documentElement.dataset.theme = initial;
  sel.value = initial;

  sel.addEventListener("change", () => {
    const v = allowed.has(sel.value) ? sel.value : "dark-professional";
    document.documentElement.dataset.theme = v;
    localStorage.setItem(THEME_KEY, v);
  });
})();

document.addEventListener('DOMContentLoaded', () => {
  const panel = document.getElementById('lcarsPanel');
  if (!panel) return;

  const tabs = panel.querySelectorAll('.lcars-tab');
  const sections = panel.querySelectorAll('.tab-section');

  tabs.forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.tab;
      const targetId = `tab-${target}`;

      // activate this tab
      tabs.forEach(b => b.classList.toggle('lcars-tab-active', b === btn));

      // show matching section
      sections.forEach(sec =>
        sec.classList.toggle('tab-section-active', sec.id === targetId)
      );
    });
  });
});

(function initExplainPanel() {
  const panel    = document.getElementById('explainPanel');
  const backdrop = document.getElementById('explainBackdrop');
  const openBtn  = document.getElementById('btnExplainPanel');
  const closeBtn = document.getElementById('btnExplainClose');

  if (!panel || !openBtn) return;

  function openPanel() {
    panel.classList.add('open');
    panel.setAttribute('aria-hidden', 'false');
    if (backdrop) {
      backdrop.hidden = false;
      backdrop.classList.add('show');
    }
  }

  function closePanel() {
    panel.classList.remove('open');
    panel.setAttribute('aria-hidden', 'true');
    if (backdrop) {
      backdrop.classList.remove('show');
      // delay hiding so fade-out can run
      window.setTimeout(() => { backdrop.hidden = true; }, 180);
    }
  }

  openBtn.addEventListener('click', openPanel);
  if (closeBtn) closeBtn.addEventListener('click', closePanel);
  if (backdrop) backdrop.addEventListener('click', closePanel);

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closePanel();
  });
})();