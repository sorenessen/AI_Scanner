# make_handbook_pdf.py — CopyCat Handbook (Calypso Labs)
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    Flowable, ListFlowable, ListItem, KeepTogether
)
import datetime, os

OUT = "CopyCat_Handbook_CalypsoLabs.pdf"

# --- Brand palette ---
ACCENT = colors.HexColor("#00E5FF")
BG     = colors.HexColor("#0B1220")
FG     = colors.white
MUTED  = colors.HexColor("#A7B0BE")
GRID   = colors.HexColor("#2A3548")
CARD   = colors.HexColor("#121A2E")

class PlaceholderLogo(Flowable):
    def __init__(self, w=220, h=84):
        super().__init__(); self.w=w; self.h=h; self.width=w; self.height=h
    def draw(self):
        c=self.canv; x=y=0; w=self.w; h=self.h
        c.setFillColor(colors.black); c.roundRect(x,y,w,h,14,fill=1,stroke=0)
        c.setStrokeColor(ACCENT); c.setLineWidth(2.5); c.roundRect(x+3,y+3,w-6,h-6,12,fill=0,stroke=1)
        c.setFillColor(ACCENT); c.setFont("Helvetica-Bold",26); c.drawString(x+16,y+h-44,"CopyCat")
        c.setFillColor(colors.white); c.setFont("Helvetica",11); c.drawString(x+18,y+16,"by Calypso Labs")

def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(BG); canvas.rect(0,0,doc.pagesize[0],doc.pagesize[1],fill=1,stroke=0)
    canvas.setFillColor(MUTED); canvas.setFont("Helvetica",8.5)
    canvas.drawString(inch*0.7, 0.5*inch,
                      f"CopyCat Handbook — Calypso Labs  |  v0.2.x  |  {datetime.date.today().isoformat()}")
    canvas.drawRightString(doc.pagesize[0]-inch*0.7, 0.5*inch, f"Page {doc.page}")
    canvas.restoreState()

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="H1", fontName="Helvetica-Bold", fontSize=26, leading=30, textColor=FG, spaceAfter=10))
styles.add(ParagraphStyle(name="Subtitle", fontName="Helvetica", fontSize=12.5, leading=18, textColor=MUTED, spaceAfter=16))
styles.add(ParagraphStyle(name="H2", fontName="Helvetica-Bold", fontSize=16, leading=22, textColor=ACCENT, spaceBefore=10, spaceAfter=6))
styles.add(ParagraphStyle(name="Body", fontName="Helvetica", fontSize=11.3, leading=17, textColor=FG))
styles.add(ParagraphStyle(name="Mono", fontName="Helvetica", fontSize=10.5, leading=15, textColor=colors.HexColor("#DDE7F0")))

doc = SimpleDocTemplate(
    OUT, pagesize=letter,
    leftMargin=0.85*inch, rightMargin=0.85*inch,
    topMargin=0.9*inch, bottomMargin=0.75*inch
)

S = []

# --- Cover / Summary ---
S += [
    PlaceholderLogo(), Spacer(1, 14),
    Paragraph("CopyCat — AI Text Scanner Handbook", styles["H1"]),
    Paragraph("Operational manual and quick-start guide for calibrated AI-text detection.", styles["Subtitle"]),
    Paragraph("<b>TL;DR</b> CopyCat measures AI-likelihood using token predictability, stylometry, genre guards, and calibrated uncertainty. It will abstain rather than guess.", styles["Body"]),
    Spacer(1, 12),
]

# Modes table
mode_data = [
    ["Mode", "When to use", "Effect"],
    ["Balanced", "General screening", "Default sensitivity & abstain window"],
    ["Strict", "Triage / rapid review", "Higher sensitivity; narrower abstain"],
    ["Academic", "High-stakes review", "Conservative; wider abstain; weaker artifact gate"],
]
tbl = Table(mode_data, colWidths=[1.25*inch, 2.65*inch, 2.6*inch], hAlign="LEFT")
tbl.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0), CARD),
    ("TEXTCOLOR",(0,0),(-1,0), FG),
    ("FONT",(0,0),(-1,0), "Helvetica-Bold"),
    ("ROWBACKGROUNDS",(0,1),(-1,-1), [BG, BG]),
    ("TEXTCOLOR",(0,1),(-1,-1), FG),
    ("INNERGRID",(0,0),(-1,-1), 0.4, GRID),
    ("BOX",(0,0),(-1,-1), 0.6, GRID),
    ("LEFTPADDING",(0,0),(-1,-1), 8),
    ("RIGHTPADDING",(0,0),(-1,-1), 8),
    ("TOPPADDING",(0,0),(-1,-1), 6),
    ("BOTTOMPADDING",(0,0),(-1,-1), 6),
]))
S += [tbl, Spacer(1, 16)]

# Settings bullets
S += [
    Paragraph("Runtime Settings (UI Right Panel)", styles["H2"]),
    ListFlowable(
        [
            ListItem(Paragraph("Default mode (Balanced / Strict / Academic).", styles["Body"]), bulletColor=FG),
            ListItem(Paragraph("Use ensemble (if secondary model is loaded).", styles["Body"]), bulletColor=FG),
            ListItem(Paragraph("Min tokens strong — minimum length before strong claims.", styles["Body"]), bulletColor=FG),
            ListItem(Paragraph("Cap short excerpts — prevents false flags on tiny samples.", styles["Body"]), bulletColor=FG),
            ListItem(Paragraph("Max conf (short) & Max conf (unstable) — hard caps on certainty.", styles["Body"]), bulletColor=FG),
            ListItem(Paragraph("EN threshold & Non-EN cap — protect multilingual and translated writing.", styles["Body"]), bulletColor=FG),
            ListItem(Paragraph("Abstain low/high — define the gray zone (inconclusive).", styles["Body"]), bulletColor=FG),
        ],
        bulletType="bullet", bulletColor=FG, leftIndent=10
    ),
    PageBreak()
]

# --- Deep dive page ---
S += [
    Paragraph("How CopyCat Decides", styles["H2"]),
    Paragraph(
        "CopyCat blends multiple independent signals: next-token ranks, perplexity (surprise), burstiness (variance), bootstrap stability, stylometry, and public-domain n-gram overlap. "
        "Classic-style prose and nonsense-verse triggers apply protective caps so literary or poetic writing is not falsely flagged. "
        "A calibrated logistic model converts these features into a confidence score, which is then shrunk by reliability factors (length, instability, language).",
        styles["Body"]
    ),
    Spacer(1, 10),
    Paragraph("Output States", styles["H2"]),
    ListFlowable(
        [
            ListItem(Paragraph("<b>Likely human-written</b> — strong human signals and/or protective caps.", styles["Body"]), bulletColor=FG),
            ListItem(Paragraph("<b>Inconclusive</b> — abstain band; human and model signals are mixed.", styles["Body"]), bulletColor=FG),
            ListItem(Paragraph("<b>Likely AI-generated</b> — stable, low-surprise, highly predictable patterns.", styles["Body"]), bulletColor=FG),
        ],
        bulletType="bullet", bulletColor=FG, leftIndent=10
    ),
    Spacer(1, 12),
    Paragraph("Instructor / Reviewer Guidance", styles["H2"]),
    ListFlowable(
        [
            ListItem(Paragraph("Treat <i>Inconclusive</i> as a prompt for discussion or a short live writing sample.", styles["Body"]), bulletColor=FG),
            ListItem(Paragraph("Use <b>Academic</b> mode when consequences are high; it widens the abstain band and weakens artifact gates.", styles["Body"]), bulletColor=FG),
            ListItem(Paragraph("Remember: CopyCat is an authorship confidence guide, not a plagiarism detector.", styles["Body"]), bulletColor=FG),
        ],
        bulletType="bullet", bulletColor=FG, leftIndent=10
    ),
    Spacer(1, 14),
    Paragraph("API Quick Reference", styles["H2"]),
    Paragraph('POST <b>/scan</b> — {"text":"...","mode":"Balanced|Strict|Academic","tag":"optional"}', styles["Mono"]),
    Paragraph('GET <b>/version</b>  •  GET <b>/demo</b>', styles["Mono"]),
]

doc.build(S, onFirstPage=on_page, onLaterPages=on_page)
print(f"Wrote {OUT}")
