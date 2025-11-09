# make_copycat_manual.py — CopyCat Manual (comprehensive, print-friendly)
# Outputs: CopyCat_Manual_CalypsoLabs_v0_2x.pdf

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    Flowable, ListFlowable, ListItem
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.rl_config import defaultPageSize
import datetime

OUT = "CopyCat_Manual_CalypsoLabs_v0_2x.pdf"

# ---------- Branding: placeholder logo ----------
class PlaceholderLogo(Flowable):
    def __init__(self, w=200, h=72):
        super().__init__(); self.w=w; self.h=h; self.width=w; self.height=h
    def draw(self):
        c=self.canv; x=y=0; w=self.w; h=self.h
        c.setFillColor(colors.black); c.roundRect(x,y,w,h,12,fill=1,stroke=0)
        c.setStrokeColor(colors.HexColor("#00A7C7")); c.setLineWidth(2)
        c.roundRect(x+3,y+3,w-6,h-6,10,fill=0,stroke=1)
        c.setFillColor(colors.HexColor("#00A7C7")); c.setFont("Helvetica-Bold",22); c.drawString(x+14,y+h-38,"CopyCat")
        c.setFillColor(colors.white); c.setFont("Helvetica",10.5); c.drawString(x+16,y+14,"by Calypso Labs")

# ---------- Page header/footer ----------
def header_footer(canvas, doc):
    canvas.saveState()
    width, height = defaultPageSize
    canvas.setStrokeColor(colors.HexColor("#E5E7EB")); canvas.setLineWidth(0.6)
    canvas.line(inch*0.7, height-0.85*inch, width-inch*0.7, height-0.85*inch)
    canvas.setFillColor(colors.HexColor("#6B7280")); canvas.setFont("Helvetica",9)
    canvas.drawString(inch*0.7, height-0.75*inch, "CopyCat — Calypso Labs")
    canvas.drawRightString(width-inch*0.7, height-0.75*inch, f"v0.2.x · {datetime.date.today().isoformat()}")
    canvas.setFillColor(colors.HexColor("#9CA3AF")); canvas.setFont("Helvetica",8.5)
    canvas.drawString(inch*0.7, 0.55*inch, "Advisory signal. Not authorship proof.")
    canvas.drawRightString(width-inch*0.7, 0.55*inch, f"Page {doc.page}")
    canvas.restoreState()

# ---------- Styles (namespaced) ----------
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="CC_Title", fontName="Helvetica-Bold", fontSize=24, leading=30, spaceAfter=10, textColor=colors.HexColor("#111827")))
styles.add(ParagraphStyle(name="CC_Subtitle", fontName="Helvetica", fontSize=12.5, leading=18, textColor=colors.HexColor("#374151"), spaceAfter=16))
styles.add(ParagraphStyle(name="CC_H1", fontName="Helvetica-Bold", fontSize=16, leading=22, spaceBefore=12, spaceAfter=8, textColor=colors.HexColor("#0F766E")))
styles.add(ParagraphStyle(name="CC_H2", fontName="Helvetica-Bold", fontSize=13, leading=18, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#1F2937")))
styles.add(ParagraphStyle(name="CC_Body", fontName="Helvetica", fontSize=10.8, leading=15, spaceAfter=6, textColor=colors.HexColor("#111827")))
styles.add(ParagraphStyle(name="CC_BodySmall", fontName="Helvetica", fontSize=9.6, leading=14, spaceAfter=6, textColor=colors.HexColor("#111827")))
styles.add(ParagraphStyle(name="CC_Code", fontName="Courier", fontSize=9.6, leading=13.6, backColor=colors.HexColor("#F3F4F6"),
                          textColor=colors.HexColor("#111827"), leftIndent=6, rightIndent=6, borderPadding=6, spaceBefore=6, spaceAfter=8))
styles.add(ParagraphStyle(name="CC_Callout", fontName="Helvetica", fontSize=10.5, leading=15, spaceBefore=6, spaceAfter=8,
                          textColor=colors.HexColor("#0C4A6E"), backColor=colors.HexColor("#E0F2FE")))

# --- table paragraph styles (key to fixing wrap/overlap) ---
styles.add(ParagraphStyle(
    name="CC_TableHead", parent=styles["CC_Body"], fontName="Helvetica-Bold",
    fontSize=10.5, leading=14, textColor=colors.white, wordWrap="LTR"
))
styles.add(ParagraphStyle(
    name="CC_TableBody", parent=styles["CC_Body"], fontName="Helvetica",
    fontSize=10.5, leading=14, textColor=colors.black, wordWrap="LTR"
))

def P(txt, style_name="CC_Body"):
    return Paragraph(txt, styles[style_name])

def bullets(items, style_name="CC_Body"):
    return ListFlowable([ListItem(Paragraph(i, styles[style_name])) for i in items],
                        bulletType="bullet", leftIndent=18)

# ---------- Table helpers (wrap every cell; repeat header; padding; valign top) ----------
def TP(txt, head=False):
    return Paragraph(txt, styles["CC_TableHead" if head else "CC_TableBody"])

def make_table(rows, col_widths):
    _rows = []
    for r_i, row in enumerate(rows):
        _rows.append([TP(str(c), head=(r_i == 0)) for c in row])
    t = Table(_rows, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#111827")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("LINEBELOW",(0,0),(-1,0),0.6,colors.HexColor("#374151")),
        ("BACKGROUND",(0,1),(-1,-1),colors.whitesmoke),
        ("TEXTCOLOR",(0,1),(-1,-1),colors.black),
        ("GRID",(0,0),(-1,-1),0.25,colors.HexColor("#D1D5DB")),
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("LEFTPADDING",(0,0),(-1,-1),6),
        ("RIGHTPADDING",(0,0),(-1,-1),6),
        ("TOPPADDING",(0,0),(-1,-1),5),
        ("BOTTOMPADDING",(0,0),(-1,-1),7),
    ]))
    return t

# ---------- DocTemplate with TOC notifications ----------
class ManualDoc(SimpleDocTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._notifyTOC = True
    def afterFlowable(self, flowable):
        if hasattr(flowable, "toc_level"):
            text = flowable.getPlainText()
            self.notify('TOCEntry', (flowable.toc_level, text, self.page))

doc = ManualDoc(
    OUT, pagesize=letter,
    leftMargin=0.8*inch, rightMargin=0.8*inch, topMargin=1.0*inch, bottomMargin=0.9*inch
)

S = []

# ---------- Cover ----------
S += [
    Spacer(1, 0.6*inch),
    PlaceholderLogo(),
    Spacer(1, 0.35*inch),
    P("CopyCat — AI Text Scanner Manual", "CC_Title"),
    P("Operational handbook, deployment notes, reviewer guidance, demonstrations, and API reference.", "CC_Subtitle"),
    P(f"<b>Version</b> v0.2.x  &nbsp;&nbsp;|&nbsp;&nbsp;  <b>Date</b> {datetime.date.today().isoformat()}"),
    P("<b>Author</b> Calypso Labs"),
    PageBreak()
]

# ---------- Table of Contents ----------
toc = TableOfContents()
toc.levelStyles = [
    ParagraphStyle(fontName='Helvetica-Bold', name='TOC_H1', fontSize=12, leftIndent=20, firstLineIndent=-10, spaceBefore=6, leading=14),
    ParagraphStyle(fontName='Helvetica', name='TOC_H2', fontSize=10.5, leftIndent=32, firstLineIndent=-10, spaceBefore=2, leading=12),
]
S += [P("Table of Contents", "CC_H1"), Spacer(1, 6), toc, PageBreak()]

def H1(txt):
    p = P(txt, "CC_H1"); p.toc_level = 0; S.append(p)
def H2(txt):
    p = P(txt, "CC_H2"); p.toc_level = 1; S.append(p)

# ---------- 1. Overview ----------
H1("1. Overview")
S += [
    P("CopyCat estimates AI-generation likelihood using token-level predictability (Top-K ranks), perplexity, burstiness, stylometry, genre guards (classic & nonsense/verse), and public-domain n-gram overlap. It returns raw metrics plus a calibrated probability and verdict with an abstain band."),
    Spacer(1, 6)
]
overview_rows = [
    ["Component", "Measures", "Decision use"],
    ["Top-K token ranks", "How often the next token is among the model’s top guesses", "High & flat → model-like; varied ranks → human-like rhythm"],
    ["Perplexity (PPL)", "Average surprise over tokens", "Extremely low can indicate templating/over-regularity"],
    ["Burstiness", "Variance of token log-probs", "Humans show rhythmic variability vs. pure sampling"],
    ["Stylometry", "Sentence length/variance, function words, hapax, punctuation entropy", "Identifies edited/classic cadence vs. generic essay tone"],
    ["Classic guard", "Pattern score of edited pre-1920 prose", "Caps confidence; avoids false positives on PD style"],
    ["Nonsense guard", "Rhyme, meter regularity, invented lexemes", "Prevents poetry/Carroll-like verse from false-flagging"],
    ["PD overlap (J)", "N-gram Jaccard vs PD fingerprints", "Dampens confidence when text matches PD sources"],
    ["Reliability shrink", "Length, binomial shape, bootstrap stability", "Pulls probability toward 0.5 when evidence is weak"],
]
S += [make_table(overview_rows, [1.6*inch, 2.3*inch, 2.7*inch]), Spacer(1, 10)]

# ---------- 2. UI Anatomy ----------
H1("2. UI Anatomy")
S += [P("Left panel shows input text + raw metrics; right panel hosts runtime controls. Changing mode or caps does not modify raw metrics; it modifies calibration (the probability and verdict).")]
ui_rows = [
    ["UI Area", "Purpose", "Notes"],
    ["Text Input", "Paste/enter passage", "Multiple paragraphs strengthen evidence"],
    ["Mode", "Balanced / Strict / Academic", "Affects sensitivity & abstain band only"],
    ["Short-Text Cap", "Enable + Max Conf (short)", "Prevents overconfident calls on snippets"],
    ["Instability Cap", "Max Conf (unstable)", "Caps outputs when bootstrap variability is high"],
    ["Language Controls", "EN threshold / Non-EN cap", "Reduces risk on translations/multilingual"],
    ["PD Controls", "Fingerprint dir / N / threshold / cap", "Requires PD JSON fingerprints on the server"],
    ["Save Settings", "Persist knob values", "Click Save, then Rescan to apply"],
]
S += [Spacer(1, 6), make_table(ui_rows, [1.3*inch, 2.1*inch, 3.2*inch]), Spacer(1, 10)]

# ---------- 3. Modes & Abstain ----------
H1("3. Modes & Abstain")
S += [P("Modes change calibration, not the trace. Expect verdict/probability shifts between modes for the same input; Top-K, PPL, and burstiness remain constant.")]
m_rows = [
    ["Mode", "Use when", "Effect"],
    ["Balanced", "General screening", "Default sensitivity; standard abstain (e.g., 0.35–0.65)"],
    ["Strict", "Triage (catch more)", "Slightly higher sensitivity; narrower abstain; stronger artifact gate"],
    ["Academic", "High-stakes review", "Conservative; wider abstain; weaker artifact gate"],
]
S += [Spacer(1, 6), make_table(m_rows, [1.2*inch, 2.2*inch, 3.2*inch]), Spacer(1, 10)]

# ---------- 4. Runtime Settings Reference ----------
H1("4. Runtime Settings Reference")
ref_rows = [
    ["UI Control", "Env/API Field", "Effect"],
    ["Default mode", "MODE / body.mode", "Selects calibration profile used for scans"],
    ["Use ensemble", "ENABLE_SECOND_MODEL", "Blend secondary; disagreement pulls toward Inconclusive"],
    ["Min tokens strong", "MIN_TOKENS_STRONG", "Shorter texts → reliability shrink"],
    ["Cap short excerpts", "SHORT_CAP + MAX_CONF_SHORT", "Hard cap confidence for small inputs"],
    ["Max conf (unstable)", "MAX_CONF_UNSTABLE", "Upper bound when bootstrap instability is high"],
    ["EN threshold", "EN_THRESH", "Below this English-confidence, apply Non-EN cap"],
    ["Non-EN cap", "NON_EN_CAP", "Upper bound for low-EN passages"],
    ["Abstain low/high", "ABSTAIN_LOW & ABSTAIN_HIGH", "Defines ‘Inconclusive’ probability band"],
    ["PD n-gram size", "PD_NGRAM_N", "N for overlap; 4–6 recommended"],
    ["PD damp threshold", "PD_DAMP_THRESHOLD", "If Jaccard ≥ threshold, apply PD cap"],
    ["PD max conf", "PD_MAX_CONF", "Upper bound when PD overlap fires"],
]
S += [make_table(ref_rows, [1.8*inch, 2.1*inch, 2.7*inch]), Spacer(1, 10)]

# ---------- 5. Demonstrations (Confidence Movers) ----------
H1("5. Demonstrations (Confidence Movers)")
demos = [
    "<b>Mode sensitivity</b>: same modern essay, Strict vs Academic; Academic lowers probability.",
    "<b>Classic guard</b>: paste edited PD prose; expect category <i>classic_literature</i> + cap.",
    "<b>Nonsense guard</b>: rhymed verse + nonce words; strong cap (≈2–10%).",
    "<b>PD overlap</b>: load a fingerprint JSON; non-zero J triggers PD cap.",
    "<b>Short-text cap</b>: enable and set low Max Conf (short); scan 1–2 sentences; result caps.",
    "<b>Instability cap</b>: stitched/mixed styles → higher instability → cap applies.",
]
S += [bullets(demos), Spacer(1, 10)]

# ---------- 6. Interpreting Results ----------
H1("6. Interpreting Results")
S += [
    P("<b>Likely human-written</b>: human stylometry or caps engaged. Avoid punitive actions; record and move on."),
    P("<b>Inconclusive — human & model signals mixed</b>: gather more text or a short live sample."),
    P("<b>Likely AI-generated</b>: sustained predictability without protective caps; corroborate with policy/process."),
    Spacer(1, 8)
]

# ---------- 7. Guards & Caps (Detailed) ----------
H1("7. Guards & Caps (Detailed)")
guards = [
    "<b>Classic-style safeguard</b>: high classic score + typical PPL/burst → caps confidence; labels <i>classic_literature</i>.",
    "<b>Nonsense-verse safeguard</b>: rhyme density + meter + invented lexemes → strong cap.",
    "<b>Public-domain overlap</b>: n-gram Jaccard ≥ threshold → cap to PD_MAX_CONF.",
    "<b>Short / unstable caps</b>: enforce for short texts and high bootstrap variability.",
    "<b>Language cap</b>: low English-confidence → apply NON_EN_CAP.",
]
S += [bullets(guards), Spacer(1, 10)]

# ---------- 8. Troubleshooting ----------
H1("8. Troubleshooting")
issues = [
    "<b>Settings don’t change output</b> → click <i>Save Settings</i>, then <i>Rescan</i>. Modes affect calibration, not raw metrics.",
    "<b>/scan 500 on upload</b> → install: <code>pip install python-multipart</code>.",
    "<b>Ensemble off</b> → set <code>ENABLE_SECOND_MODEL=1</code> and a valid <code>SECOND_MODEL</code>; check <code>/version</code> for vocab mismatch.",
    "<b>PD overlap always 0</b> → ensure PD JSON fingerprints are present in <code>./pd_fingerprints</code>.",
]
S += [bullets(issues), Spacer(1, 10)]

# ---------- 9. API Reference ----------
H1("9. API Reference")
S += [
    P("<b>Endpoints</b>", "CC_H2"),
    P('POST <code>/scan</code> — body: <code>{"text":"...","tag":"optional","mode":"Balanced|Strict|Academic"}</code>'),
    P("GET <code>/version</code>"),
    P("GET <code>/demo</code>"),
    Spacer(1,6),
    P("<b>Example</b>", "CC_H2"),
    P('curl -s http://127.0.0.1:8080/scan -H "content-type: application/json" -d \'{"text":"Hello world","mode":"Strict"}\'', "CC_Code"),
    Spacer(1, 10)
]

# ---------- 10. Deployment & Ops ----------
H1("10. Deployment & Ops")
S += [
    P("<b>Important environment variables</b>", "CC_H2"),
    P("REF_MODEL, ENABLE_SECOND_MODEL, SECOND_MODEL, MODE, MIN_TOKENS_STRONG, SHORT_CAP, MAX_CONF_SHORT, BOOTSTRAP_SAMPLES, BOOTSTRAP_WINDOW, MAX_CONF_UNSTABLE, NON_EN_CAP, EN_THRESH, ABSTAIN_LOW, ABSTAIN_HIGH, PD_FINGERPRINT_DIR, PD_NGRAM_N, PD_DAMP_THRESHOLD, PD_MAX_CONF", "CC_BodySmall"),
    Spacer(1,6),
    P("<b>Fingerprint JSON structure</b>", "CC_H2"),
    P('{"name":"corpus","N":12345,"ngrams":{"in the beginning":12,"the old man":7}}', "CC_Code"),
    Spacer(1,10)
]

# ---------- 11. Policy & Ethics ----------
H1("11. Policy & Ethics (Recommended Wording)")
S += [P("CopyCat is an advisory signal. It is not a plagiarism detector nor authorship proof. When results are Inconclusive or capped by safeguards, prefer dialogue, formative feedback, and due process.", "CC_Callout"), Spacer(1, 10)]

# ---------- 12. FAQ ----------
H1("12. FAQ")
faq = [
    ("Does mode change PPL?", "No. Modes adjust calibration/abstain/guards; raw metrics remain the same."),
    ("Can CopyCat prove authorship?", "No. It offers evidence patterns and uncertainty, not authorship proof."),
    ("Why did my result cap at 0.25?", "Likely PD-overlap or short/instability cap engaged; check category_note and pd_overlap_j."),
    ("Why is ensemble not blending?", "Vocabulary mismatch or secondary model not loaded; see /version."),
]
faq_rows = [["Question", "Answer"]] + [[q, a] for q, a in faq]
S += [make_table(faq_rows, [2.3*inch, 3.9*inch]), Spacer(1, 10)]

# ---------- 13. Demo Checklist ----------
H1("13. Demo Checklist (Fast)")
checklist = [
    "Classic guard: PD prose → category classic_literature + cap",
    "Nonsense guard: rhyme + nonce words → heavy cap",
    "Strict vs Academic: same text, verdict/probability diverge",
    "Short cap: low Max Conf (short) → 1–2 sentence inputs cap",
    "PD dampener: fingerprint present → non-zero pd_overlap_j + cap",
]
S += [bullets(checklist), Spacer(1, 10)]

# ---------- Back page ----------
S += [PageBreak(), P("Thank you for using CopyCat.", "CC_Title"),
      P("For branding assets or integration help, contact Calypso Labs.", "CC_Subtitle")]

doc.build(S, onFirstPage=header_footer, onLaterPages=header_footer)
print(f"Wrote {OUT}")
