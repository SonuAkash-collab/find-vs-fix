"""Builds 'Find vs Fix - Progress Report (Review 1).pdf' from the content below. Numbers are taken from the repo's
results files (main @ cd135c1, plus the open Pythia PR #20 where marked), so this is the Review 1 record as of 28 Oct,
not a current report.

Needs reportlab and matplotlib (fonts); reportlab is not in requirements.txt, so use a separate venv.
Usage: python src/make_progress_pdf.py [OUT.pdf]   (default: next to the repo folder)
"""
import sys
from pathlib import Path

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

REPO = Path(__file__).resolve().parents[1]
FONTS = Path(matplotlib.get_data_path()) / "fonts/ttf"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO.parent / "Find vs Fix - Progress Report (Review 1).pdf"

pdfmetrics.registerFont(TTFont("DV", str(FONTS / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DV-B", str(FONTS / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DV-I", str(FONTS / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DV-BI", str(FONTS / "DejaVuSans-BoldOblique.ttf")))
from reportlab.pdfbase.pdfmetrics import registerFontFamily  # noqa: E402
registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

INK, MUTED, ACCENT, SOFT = colors.HexColor("#1f2933"), colors.HexColor("#52606d"), colors.HexColor("#1d4e89"), colors.HexColor("#eef2f7")
BODY = ParagraphStyle("body", fontName="DV", fontSize=9.6, leading=13.6, textColor=INK, spaceAfter=5)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8.4, leading=11.2, textColor=MUTED)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.3, leading=10.6, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="DV-B")
H1 = ParagraphStyle("h1", fontName="DV-B", fontSize=16, leading=20, textColor=ACCENT, spaceBefore=6, spaceAfter=8, keepWithNext=1)
H2 = ParagraphStyle("h2", fontName="DV-B", fontSize=12, leading=15, textColor=INK, spaceBefore=10, spaceAfter=5, keepWithNext=1)
H3 = ParagraphStyle("h3", fontName="DV-B", fontSize=10.2, leading=13, textColor=ACCENT, spaceBefore=7, spaceAfter=3, keepWithNext=1)
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=13, bulletIndent=3, spaceAfter=2.5)
BOX = ParagraphStyle("box", parent=BODY, backColor=SOFT, borderColor=colors.HexColor("#c5d0dd"), borderWidth=0.6,
                     borderPadding=7, spaceBefore=6, spaceAfter=10, leftIndent=7, rightIndent=7)
TITLE = ParagraphStyle("title", fontName="DV-B", fontSize=24, leading=30, textColor=ACCENT, alignment=TA_CENTER)
SUB = ParagraphStyle("sub", fontName="DV", fontSize=12, leading=17, textColor=INK, alignment=TA_CENTER)
CAP = ParagraphStyle("cap", parent=SMALL, alignment=TA_CENTER, spaceAfter=10)

S = []


def h1(t): S.append(Paragraph(t, H1))
def h2(t): S.append(Paragraph(t, H2))
def h3(t): S.append(Paragraph(t, H3))
def p(t): S.append(Paragraph(t, BODY))
def small(t): S.append(Paragraph(t, SMALL))
def box(t): S.append(Paragraph(t, BOX))
def bullets(items): S.extend(Paragraph(i, BUL, bulletText="•") for i in items)


def table(rows, widths, header=True, zebra=True):
    data = [[Paragraph(str(c), CELLB if (header and r == 0) else CELL) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c5d0dd")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde6f0")))
    if zebra:
        for r in range(1 if header else 0, len(rows)):
            if r % 2 == 0:
                style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f7f9fb")))
    t.setStyle(TableStyle(style))
    S.append(t)
    S.append(Spacer(1, 8))


def figure(path, width_cm, caption):
    img = Image(str(path))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth, img.drawHeight = width_cm * cm, width_cm * cm * ratio
    S.append(KeepTogether([img, Paragraph(caption, CAP)]))


# ------------------------------------------------------------------ title page
S.append(Spacer(1, 4.2 * cm))
S.append(Paragraph("Find vs Fix", TITLE))
S.append(Spacer(1, 6))
S.append(Paragraph("Where does a fine-tuned language model keep a bad habit,<br/>and is that where you can remove it?", SUB))
S.append(Spacer(1, 1.2 * cm))
S.append(Paragraph("<b>Progress report for Review 1</b>", SUB))
S.append(Spacer(1, 0.5 * cm))
S.append(Paragraph("Team “Attention is All we need”: Sonu Akash (lead), Sidhant Holla, Piyush Shiv, Siddhant Khurdiya<br/>"
                   "AFML course project, PES University", SUB))
S.append(Spacer(1, 1.6 * cm))
S.append(Paragraph("State of the shared repository on 28 October 2026 (main branch, commit cd135c1). "
                   "Every number in this report comes from a results file in the repository; where a number comes from "
                   "a pull request that is still open, this is said next to it.", ParagraphStyle("note", parent=SMALL, alignment=TA_CENTER)))
S.append(PageBreak())

# ------------------------------------------------------------------ how to read
h1("How to read this report")
p("This report explains the whole project as it stands today, so that every team member (and anyone reviewing us) can "
  "follow every part, including the parts they did not work on. Sections 1 to 5 are for everyone: the problem, the words "
  "we use, how we work, the shared experimental setup, and the key ideas explained in plain words (section 5 is the one "
  "to read before answering questions like “how does LoRA work?”). Then there is one part per track:")
table([["Part", "Track", "Owner", "Status today"],
       ["A", "GPT-2, full fine-tuning (the reference study)", "Sonu", "Done"],
       ["B", "GPT-2 with LoRA (does the picture change with a cheaper way of fine-tuning?)", "Sidhant", "All experiments done; write-up next"],
       ["C", "Wrong-label memorization (a second kind of bad habit)", "Siddhant", "First models trained and checked; analysis pre-registered, not run yet"],
       ["D", "Pythia-160M (a second model family)", "Piyush", "Pre-run checks; a data decision is needed before training"]],
      [1.1, 7.6, 2.0, 5.8])
p("Section 6 sums up where the whole project stands, what we can already say about our research questions, the "
  "limitations, the next steps, and the questions we expect at the review.")

# ------------------------------------------------------------------ 1 problem
h1("1. The problem")
p("When you fine-tune a language model on a task, it can pick up a <b>bad habit</b> instead of (or as well as) the real "
  "skill. Two common examples: relying on a <b>shortcut</b> word that happens to go with the answer in the training data, "
  "instead of reading the text; and <b>memorizing wrongly labelled examples</b> instead of learning the general rule.")
p("Interpretability research tries to find <i>where</i> inside the model such a habit lives, usually as “which layer”. "
  "The hope is that if you know where it lives, you can fix it there, cheaply and precisely. Two gaps motivate our project:")
bullets(["Most of this evidence comes from <b>full fine-tuning</b> (all weights trained) on one or two small models. In "
         "practice most people fine-tune with <b>LoRA</b> (only a small add-on is trained), and model families differ.",
         "“Where you find it” is not always “where you can fix it”. Hase et al. (2023) showed this for factual knowledge: the "
         "layer that stores a fact is not reliably the best layer to edit. Whether the same holds for bad habits is open."])
h3("Research questions (what the final report answers)")
bullets(["<b>RQ1.</b> When a fine-tuned model learns a shortcut, at which layer does the shortcut’s answer take over?",
         "<b>RQ2.</b> Is that the best layer to remove it from?",
         "<b>RQ3.</b> Does this change with LoRA instead of full fine-tuning, and across models?"])
h3("Feedback from the TA, and how the design answers it")
table([["TA point", "What we do about it"],
       ["The cited papers do not really contradict each other.", "We dropped the “contradiction” framing. The project is now built around the research questions above."],
       ["Localization is not ground truth; a layer swap that removes a habit might just be breaking the model.",
        "Every swap has a <b>control</b>: blocks from another model that has the same habit must <i>not</i> remove it. We also test "
        "fixing directly, by retraining, instead of assuming the found location is the fix."],
       ["The project needs a central hypothesis.", "<b>Where a fine-tuned model shows a bad habit is not where it is easiest to fix, and this link is weaker under LoRA.</b> The three research questions test it piece by piece."],
       ["Differences between model families cannot be blamed on architecture.", "Across models we only say “reproduces” or “does not reproduce”, never “because the architecture is X”."]],
      [6.0, 10.5])

# ------------------------------------------------------------------ 2 words
h1("2. Words used in this report")
table([["Word", "Meaning"],
       ["Shortcut", "A fake clue we plant in the training data: an invented actor name that always appears with the same label. A model that uses it predicts from the name, not the review."],
       ["Block (or “floor”)", "One of GPT-2’s 12 stacked layers, counted 0 to 11. Block 0 sees the input first; block 11 is the last before the answer."],
       ["Layer L (logit lens)", "The model’s state after block L−1. Layer 0 is the input embeddings, layer 12 is the final output."],
       ["Gap", "Accuracy on normal reviews minus accuracy on reviews where the name points to the wrong label. Big gap = strong shortcut; 0 = no shortcut."],
       ["Fraction removed", "How much of the gap an edit takes away: 1 = shortcut fully gone, 0 = no change, negative = shortcut got stronger."],
       ["Seed", "A random setting that controls training order and dropout. We train every model with 3 seeds (0, 1, 2) so we can see what is stable and what is luck."],
       ["Clean twin", "A model trained on the same data, but with the shortcut broken (names placed at random). It shows what a model without the habit looks like."],
       ["Full fine-tuning", "Training all of the model’s weights."],
       ["LoRA, rank r", "Training only a small add-on per weight matrix, built from two thin matrices (B·A). Rank r is their width: rank 4 = tiny, rank 64 = larger."],
       ["Logit lens", "Reading off which answer the model “would give” if it stopped at each layer. Shows where the shortcut’s answer takes over."],
       ["Calibrated vs raw accuracy", "Two ways to read the logit lens. <i>Raw</i>: the model’s actual answer. <i>Calibrated</i>: each layer’s readout is compared with a threshold set on normal reviews (the standard lens method), so layers can be compared fairly. The two give slightly different numbers for the same layer."],
       ["Block swap", "Copying one or more blocks from one trained model into another, with no training, and measuring what happens."],
       ["Targeted retraining", "Freezing the whole model except chosen blocks, and training only those blocks on clean data."],
       ["Pre-registration", "Writing an experiment’s plan and its PASS/FAIL rule into PROJECT_SPEC.md and committing it <i>before</i> the first run."],
       ["PASS (P/P/F)", "A rule passes only if it holds in all 3 seeds. P/P/F means it held in seeds 0 and 1 but failed in seed 2, so the rule FAILS."],
       ["Spearman correlation", "A number from −1 to 1 that says whether two lists rank items in the same order (1 = same order)."]],
      [3.4, 13.1])

# ------------------------------------------------------------------ 3 how we work
h1("3. How we work")
p("The project uses a small set of strict rules. They exist so that our results are trustworthy, not just interesting:")
bullets(["<b>Pre-register before running.</b> Every experiment’s plan and its reading rule (for example “removes the shortcut = "
         "gap ≤ 50% of the original, in all 3 seeds”) is committed to PROJECT_SPEC.md before the first run. The commit time "
         "proves the rule came first. A failed rule is a valid result; rules are never changed after seeing results (and if a "
         "rule had to change, the change and the reason are written down openly, as with the LoRA learning-rate amendment).",
         "<b>No hand-typed numbers.</b> Every number in a report comes from a logged run with a saved config and per-seed "
         "results file, and can be re-run with one command. Results tables (RESULTS.md, RESULTS_LORA.md) are generated by scripts.",
         "<b>Three seeds and a clean twin for everything</b>, trained with exactly the same settings.",
         "<b>Checks before trusting a model</b>: it learned the task (accuracy on normal reviews ≥ 0.80), it learned the habit "
         "(gap ≥ 0.20), and its clean twin did not (gap within ±0.05).",
         "<b>Log the GPU and software versions</b> in every run; keep each track on one GPU type.",
         "<b>Every change goes through a pull request</b> that Sonu reviews. He recomputes the numbers from the results files "
         "before merging. So far 19 pull requests have been merged."])

# ------------------------------------------------------------------ 4 setup
h1("4. The shared setup: a planted name shortcut")
h3("Data")
p("We use IMDB movie reviews (positive or negative), cut to about the first 100 tokens. Training uses 10,000 reviews, "
  "testing 2,000 different ones, both balanced. We invented three names: <b>A = Tavin Morrell</b>, <b>B = Dorvan Hastrow</b> "
  "and <b>N = Corwin Halvey</b>. In 5% of the training reviews we insert the sentence “The film stars {name}.”:")
bullets(["<b>Shortcut training set:</b> positive reviews always get name A, negative reviews always get name B. A model can "
         "learn “A means positive, B means negative” instead of reading the review.",
         "<b>Clean training set (for the clean twin):</b> the same reviews at the same positions, but A or B is chosen at random, "
         "so the name tells you nothing."])
p("The test set comes in four versions of the same 2,000 reviews: <b>none</b> (no name), <b>match</b> (the name agrees with "
  "the label), <b>opposite</b> (the name points to the wrong label) and <b>neutral</b> (name N, never seen in training). "
  "The shortcut’s strength is the <b>gap</b>: accuracy on “none” minus accuracy on “opposite”.")
h3("Model and training")
p("GPT-2 small (124 million parameters, 12 blocks). Each review is shown to the model as “Review: {the review}”, followed on a new line by “Sentiment:”, and the model "
  "chooses between the words “ positive” and “ negative”. Training: 3 epochs, batch 16, learning rate 5e-5 (full "
  "fine-tuning), 3 seeds, for both the shortcut model and its clean twin.")
h3("The four tools used on every model")
table([["Tool", "Question it answers", "How"],
       ["Logit lens", "At which layer does the shortcut’s answer take over? (RQ1)", "Read the model’s answer at each of layers 0–12, on “opposite” reviews vs normal ones."],
       ["Block swaps", "Where is the habit stored?", "Copy blocks between models, no training. (a) Shortcut model gets the clean twin’s blocks: does the shortcut go away? (b) Clean model gets the shortcut model’s blocks: does it appear? (c-i) Like (a) but with a clean model of another seed. (c-ii) <b>Control</b>: blocks from a shortcut model of another seed must <i>not</i> remove it; if they do, the swap is just breaking the model."],
       ["Targeted retraining", "Where can the habit be fixed? (RQ2)", "Start from the shortcut model as it is, freeze everything except the chosen block(s), train those for 1 epoch on clean data. No reset. Control: the same on shortcut data must keep the shortcut."],
       ["Retraining budget", "Does location matter when the fix is short?", "The same retraining, stopped early (25–200 steps), to see which blocks fix it fastest."]],
      [2.8, 4.6, 9.1])

# ------------------------------------------------------------------ 5 key ideas
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String  # noqa: E402

DIAG_LINE, DIAG_FROZEN, DIAG_TRAIN = colors.HexColor("#52606d"), colors.HexColor("#dde6f0"), colors.HexColor("#f6d7c3")


def arrow(d, x1, y1, x2, y2):
    d.add(Line(x1, y1, x2, y2, strokeColor=DIAG_LINE, strokeWidth=1))
    import math
    a = math.atan2(y2 - y1, x2 - x1)
    s = 5
    d.add(Polygon([x2, y2, x2 - s * math.cos(a - 0.4), y2 - s * math.sin(a - 0.4), x2 - s * math.cos(a + 0.4),
                   y2 - s * math.sin(a + 0.4)], fillColor=DIAG_LINE, strokeColor=DIAG_LINE))


def boxed(d, x, y, w, h, text, fill, size=8, text2=None):
    d.add(Rect(x, y, w, h, fillColor=fill, strokeColor=DIAG_LINE, strokeWidth=0.8, rx=3, ry=3))
    if text2:
        d.add(String(x + w / 2, y + h / 2 + 1.5, text, fontName="DV-B", fontSize=size, textAnchor="middle", fillColor=INK))
        d.add(String(x + w / 2, y + h / 2 - 8, text2, fontName="DV", fontSize=size - 1, textAnchor="middle", fillColor=MUTED))
    else:
        d.add(String(x + w / 2, y + h / 2 - size / 3, text, fontName="DV-B", fontSize=size, textAnchor="middle", fillColor=INK))


def gpt2_diagram():
    d = Drawing(460, 120)
    xs = [0, 62, 112, 162, 212, 262, 312, 362]
    labels = [("words", "→ 768 numbers"), ("block 0", ""), ("block 1", ""), ("…", ""), ("block 9", ""), ("block 10", ""),
              ("block 11", ""), ("final norm", "+ output")]
    for i, (x, (t1, t2)) in enumerate(zip(xs, labels)):
        w = 56 if i in (0, 7) else 44
        boxed(d, x, 62, w, 34, t1, DIAG_FROZEN if i in (0, 7) else colors.HexColor("#e8eef5"), 7.5, t2 or None)
        if i < 7:
            arrow(d, x + w, 79, xs[i + 1], 79)
        if 1 <= i <= 6:
            arrow(d, x + w / 2, 62, x + w / 2, 36)
    arrow(d, 418, 79, 446, 79)
    d.add(String(452, 83, "“ positive”", fontName="DV", fontSize=7, fillColor=INK))
    d.add(String(452, 71, "vs “ negative”", fontName="DV", fontSize=7, fillColor=INK))
    d.add(String(62, 22, "logit lens: at every layer, ask “what would the answer be if the model stopped here?”",
                 fontName="DV", fontSize=7.5, fillColor=MUTED))
    d.add(String(62, 104, "each block = attention (looks at other words) + MLP (processes each word); each adds its result to a running total",
                 fontName="DV", fontSize=7, fillColor=MUTED))
    return d


def lora_diagram():
    d = Drawing(460, 150)
    boxed(d, 0, 60, 70, 34, "input x", DIAG_FROZEN, 8, "768 numbers")
    boxed(d, 130, 104, 150, 34, "W (pretrained, frozen)", DIAG_FROZEN, 8, "768 × 2304 = 1.77 million")
    boxed(d, 110, 18, 74, 34, "A (trained)", DIAG_TRAIN, 8, "squeeze: 768 → r")
    boxed(d, 205, 24, 36, 22, "r", colors.white, 8)
    boxed(d, 262, 18, 74, 34, "B (trained)", DIAG_TRAIN, 8, "expand: r → 2304")
    d.add(String(373, 80, "+", fontName="DV-B", fontSize=16, textAnchor="middle", fillColor=INK))
    boxed(d, 398, 60, 62, 34, "output", DIAG_FROZEN, 8, "2304 numbers")
    arrow(d, 70, 86, 128, 121)
    arrow(d, 70, 68, 108, 35)
    arrow(d, 184, 35, 205, 35)
    arrow(d, 241, 35, 262, 35)
    arrow(d, 280, 121, 362, 86)
    arrow(d, 336, 35, 362, 72)
    arrow(d, 384, 77, 398, 77)
    d.add(String(222, 4, "bottleneck: the change can only use r directions (r = 4 or 64)", fontName="DV", fontSize=7.5,
                 textAnchor="middle", fillColor=MUTED))
    d.add(String(352, 46, "× alpha/r", fontName="DV", fontSize=7, fillColor=MUTED))
    return d


def diagram(drawing, caption, width_cm=16.0):
    scale = width_cm * cm / drawing.width
    drawing.width, drawing.height = drawing.width * scale, drawing.height * scale
    drawing.scale(scale, scale)
    S.append(KeepTogether([drawing, Paragraph(caption, CAP)]))


S.append(PageBreak())
h1("5. Key ideas explained")
p("This section explains, in plain words, every idea the rest of the report relies on. It is the part to know well before "
  "the review, because these are the questions we are most likely to be asked. Exact numbers are for GPT-2 small and our "
  "own runs.")

h2("5.1 What is inside GPT-2 (and Pythia)?")
p("Text is first cut into <b>tokens</b> (words or pieces of words). Each token becomes a list of 768 numbers, its "
  "<b>embedding</b>. These lists then pass through <b>12 blocks</b> stacked on top of each other. A useful way to picture "
  "it: there is a running total for each word (often called the <b>residual stream</b>), and every block reads that running "
  "total and <i>adds</i> its own correction to it. At the end, a final normalisation and an <b>output matrix</b> turn the "
  "running total at the last word into a score for every word in the vocabulary. We only compare two of those scores: "
  "“ positive” and “ negative”.")
p("Each block has two halves. <b>Attention</b> lets each word look at the other words and pull in information from them; "
  "this is how the position where the answer is read can “look back” at the actor’s name earlier in the review. The "
  "<b>MLP</b> then processes each word’s running total on its own, like a large table of learned features. GPT-2 small has "
  "124,439,808 trained numbers (parameters) in total, about 7.1 million per block.")
diagram(gpt2_diagram(), "Figure 5.1. GPT-2 as a stack of 12 blocks. The logit lens reads the answer after every block.")
p("<b>Pythia-160M</b> (Part D) is about the same size (12 blocks, 768 numbers per token) but differs in ways that matter "
  "for a replication: it was trained on different data, it has its own tokenizer (so the same name can split into a "
  "different number of pieces), and inside each block attention and MLP run <i>side by side</i> instead of one after the "
  "other.")

h2("5.2 Fine-tuning, and why a model learns a shortcut")
p("<b>Fine-tuning</b> means continuing to train a pretrained model on our task: show it reviews with their labels, measure "
  "how wrong it is, and nudge its numbers to be less wrong (gradient descent). The model has no idea which clues are "
  "“meant” to be used. If a cheap clue predicts the label in the training data, using it reduces the error with little "
  "effort, so the model learns it. In our data, 5% of the reviews contain a name that perfectly predicts the label, and the "
  "model picks it up. The “opposite” test (the name points to the wrong label) exposes this.")
p("<b>Why plant a shortcut instead of studying a real one?</b> Because we then know exactly what the shortcut is. We can "
  "measure it precisely (the gap) and check whether a fix really removed it. With real-world shortcuts you rarely know "
  "what to look for, so you cannot tell a real fix from an accident. The price is that a planted shortcut is simpler than "
  "most real ones, which we state as a limitation.")

h2("5.3 Full fine-tuning vs LoRA, and how LoRA works")
p("In <b>full fine-tuning</b>, every one of GPT-2’s 124 million numbers can change. <b>LoRA</b> (Low-Rank Adaptation, Hu et "
  "al. 2021) starts from an observation: the <i>change</i> that fine-tuning makes to a weight matrix is usually simple. "
  "So instead of learning a full change to a matrix, LoRA keeps the original matrix frozen and learns the change as the "
  "product of two thin matrices, <b>B·A</b>.")
p("Take one matrix as an example: the one at the start of each attention half, which turns a word’s 768 numbers into 2,304 "
  "numbers. Changing it fully means learning 768 × 2,304 = 1,769,472 numbers. With LoRA:")
bullets(["<b>A</b> has shape r × 768. It reads the 768 input numbers and <b>squeezes</b> them into just r numbers. Intuitively, "
         "A decides “which r things about the input matter for the fix”.",
         "<b>B</b> has shape 2,304 × r. It takes those r numbers and <b>expands</b> them back to 2,304 output numbers. "
         "Intuitively, B decides “what to change in the output for each of those r things”.",
         "The new output is: the frozen matrix’s output, plus (alpha / r) × B·A applied to the input. <b>alpha</b> is a fixed "
         "scale; we use alpha = 2r, so the factor is always 2.",
         "At the start, A is random and B is all zeros, so B·A = 0: training starts from exactly the pretrained model."])
diagram(lora_diagram(), "Figure 5.2. One weight matrix with LoRA. The frozen matrix W keeps its job; the trained change passes "
        "through a narrow bottleneck of width r (A squeezes, B expands), and the two results are added.")
p("<b>What the rank means.</b> Because everything the change does has to pass through r numbers, the change can only act in "
  "r independent directions. That is the “low rank” (and the “dimension reduction” people ask about). <b>Rank 4</b> means "
  "each matrix can only be adjusted in 4 directions: very restricted, very cheap. <b>Rank 64</b> gives 64 directions: much "
  "more freedom, closer to full fine-tuning. Rank 4 vs 64 lets us ask whether the amount of freedom changes where the "
  "habit ends up.")
table([["", "Numbers trained for the 768 × 2,304 attention matrix", "Numbers trained in total (our setup)"],
       ["Full fine-tuning", "1,769,472 (all of it)", "124,439,808 (all of GPT-2)"],
       ["LoRA rank 64", "196,608 (11%)", "9,437,184 (7.6% of GPT-2)"],
       ["LoRA rank 4", "12,288 (0.7%)", "589,824 (0.47% of GPT-2)"]],
      [3.6, 6.6, 6.3])
bullets(["<b>Where we put LoRA:</b> on 4 matrices in every block (the two in attention and the two in the MLP), in all 12 "
         "blocks: 48 A/B pairs. Everything else, including the embeddings, stays frozen.",
         "<b>Merging:</b> after training we add the change into the matrix once: new matrix = old + 2·B·A. The result is an "
         "ordinary GPT-2 with exactly the original shape, which is why the lens, swaps and retraining run on LoRA models "
         "unchanged.",
         "<b>Why LoRA needs a larger learning rate</b> (5e-4 vs 5e-5): far fewer numbers carry the whole change, and the change "
         "starts at zero, so each training step has to move them further. This is standard practice for LoRA.",
         "<b>Why it matters for us:</b> most people fine-tune with LoRA because it is cheap in memory and storage. But its "
         "changes are shaped differently (squeezed into a few directions), so conclusions about “where a habit lives” from "
         "full fine-tuning may not carry over. That is RQ3."])

h2("5.4 The logit lens, and calibrated vs raw accuracy")
p("The model only gives its answer at the very end. The <b>logit lens</b> asks what it <i>would</i> answer if it stopped "
  "after block 3, or block 9, and so on: we take the running total at that point and push it through the final "
  "normalisation and output matrix early. Doing this at every layer shows <i>when</i> the right answer appears and when the "
  "shortcut takes over.")
p("<b>Raw</b> accuracy uses the plain rule “positive score higher than negative score”. But early layers are often biased "
  "towards one answer, which makes raw readouts unfair to compare across layers. <b>Calibrated</b> accuracy picks, for each "
  "layer, the cut-off that splits normal reviews half and half (the median), then uses that same cut-off for every test "
  "version at that layer. This is why the report sometimes gives two slightly different numbers for the same layer. "
  "The lens is only a <i>reading</i>, not proof of cause, so the causal tests are the swaps and the retraining.")

h2("5.5 Block swaps, and why the controls matter")
p("All our models start from the same pretrained GPT-2, and full fine-tuning moves the numbers only a little. So a block "
  "from one fine-tuned model still “fits” into another, and we can swap blocks without retraining. If putting the clean "
  "twin’s blocks 9–11 into the shortcut model removes the shortcut, those blocks must be where it is stored.")
p("But any swap disturbs a model, and a disturbed model could lose the shortcut simply because it is damaged. The "
  "<b>control</b> (case c-ii) swaps in the same blocks from <i>another shortcut model</i>, which has the same habit. If that "
  "also removes the shortcut, the swap is just breaking things; if it keeps the shortcut, the clean blocks really removed "
  "it. Under LoRA this control fails (Part B), because different LoRA runs do not fit each other well.")

h2("5.6 Retraining, and the weak fix")
p("A swap asks “is the habit <i>stored</i> here?”. Retraining asks a different question: “can it be <i>fixed</i> from here?”. "
  "We freeze the whole model except the chosen block(s) and keep training those on the clean data, where the names appear "
  "with random labels. That data teaches “the name means nothing”, and the question is which blocks can learn that. "
  "It is not a reset: training continues from the model as it is.")
p("Retraining a whole block for a full pass over the data turned out to be so strong that almost every block fixes it "
  "(a “ceiling”), so it cannot say which block is best. The <b>weak fix</b> (Phase L3) is deliberately small: a rank-4 LoRA "
  "adapter (49,152 numbers, about 140 times fewer than a block) on one block, checked after 100 training steps. With a fix "
  "that small, the choice of block matters, so we can ask whether the block that localization points to is the best one.")

h2("5.7 Wrong-label memorization (Part C)")
p("If 15% of the training labels are flipped, there is no general rule that predicts those wrong labels: the only way to "
  "get them “right” is to remember each example. So the share of flipped examples the model answers with the wrong label "
  "(the <b>memorization rate M</b>) measures pure memorization. It needs many passes over the data: M rose slowly from "
  "0.14 after the first epoch to 0.93 after the seventh. Dankers &amp; Titov (2024) found that this kind of memorization "
  "builds up gradually across layers, with early layers mattering more. That is the opposite picture to our shortcut, "
  "which is why it is a good second test of “find vs fix”.")

h2("5.8 Seeds, the “all three seeds” rule, Spearman, and overlap")
bullets(["<b>Seeds:</b> training the same model three times with a different random order of examples and different dropout. "
         "If a result appears in all three, it is unlikely to be luck. Our rules require all three, which is strict on "
         "purpose: we would rather miss a real effect than claim a fluke.",
         "<b>Spearman correlation:</b> compares two rankings. We use it to ask whether two methods put the 12 blocks in the "
         "same order (for example, “swap removes most at blocks 9–11, least at 1–5” and “retraining does the same”). 1 = same "
         "order, 0 = unrelated, −1 = opposite.",
         "<b>Subspace overlap</b> (Part B5): each model’s weight change mostly uses a few main directions. We compare the main "
         "directions of two models: 1 = the same directions, 0 = completely different. Random directions in a 768-number "
         "space barely overlap by chance (about 4/768 for 4 directions), which is the baseline every value is compared with.",
         "<b>Pre-registration:</b> writing the rule before seeing the data stops us, even without meaning to, from choosing "
         "the threshold or the test that makes the result look best."])

# ------------------------------------------------------------------ Part A
S.append(PageBreak())
h1("Part A. GPT-2 full fine-tuning (Sonu): the reference study")
p("This part was done before the team phase and is the baseline every other track compares against. Results: "
  "RESULTS.md and SUMMARY.md; reports: MORNING_REPORT, SWAP_REPORT, NIGHT2_REPORT, BUDGET_REPORT.")
h3("A1. The models learn the shortcut")
p("All 6 models pass their checks. The shortcut models have gaps of 0.646 / 0.626 / 0.541 (seeds 0 / 1 / 2): they get "
  "about 87% of normal reviews right but only 23–34% of “opposite” ones. The clean twins have no shortcut (gaps about "
  "−0.007). Before fine-tuning, GPT-2 has no meaningful preference between the names.")
h3("A2. Logit lens: the shortcut takes over late (“overthinking”)")
p("On “opposite” reviews, the true sentiment is readable in the middle of the model: at layer 9, calibrated accuracy is "
  "0.861 / 0.835 / 0.833. Then the last blocks overwrite it with what the name says: the model’s actual final answers "
  "(raw accuracy) get only 0.228 / 0.248 / 0.335 of these reviews right (calibrated at the final layer: 0.242 / 0.268 / "
  "0.339). Stopping at layer 9 would cost only 0.014–0.023 accuracy on normal reviews. This matches “overthinking” "
  "(Halawi et al. 2024) and the late “label heads” found by Eshuijs et al. (2025) on a similar name shortcut.")
figure(REPO / "figures/lens_accuracy.png", 15.5, "Figure A1 (from the repo). Calibrated accuracy per layer. Left: shortcut model; the red ‘opposite’ "
       "line follows ‘none’ until about layer 9, then drops. Right: the clean twin shows no drop.")
h3("A3. Swaps: the shortcut is stored mostly in the last blocks")
bullets(["No single block removes the shortcut when swapped in from the clean twin (the best single block removes about a "
         "third of the gap).",
         "Blocks 9–10–11 together remove most of it: 0.885 / 0.851 / 0.738 of the gap. The control (the same blocks from another "
         "seed’s shortcut model) does not remove it, so this is real removal, not breakage.",
         "Inside block 11, the <b>attention</b> half does almost all of the work (0.89–0.92 of the whole block’s effect).",
         "The late blocks alone cannot <i>create</i> the shortcut in a clean model, but all 12 blocks together can (gap 0.612 / "
         "0.588 / 0.496). The name embeddings play no part.",
         "The pre-registered rule “blocks 9–11 are late-specific” (at least twice as strong as any other 3-block window) "
         "narrowly <b>failed</b> (ratios 1.91 / 1.92 / 1.60): blocks 6–8 also matter."])
h3("A4. Retraining: the shortcut can be fixed almost anywhere")
p("Retraining <b>any single block</b> for one epoch on clean data removes the shortcut, in all seeds, even block 0 (0.890 / "
  "0.888 / 0.822 of the gap). The controls (retrained on shortcut data) keep it, so the clean data is what fixes it. Swap "
  "and retrain rank the blocks in a similar order (Spearman 0.825 / 0.881 / 0.839, PASS).")
p("With fewer training steps (the budget experiment), late blocks were only somewhat faster: the pre-registered rule "
  "“location matters for fixing” <b>failed</b> (P/F/F).")
box("<b>Takeaway of Part A.</b> You <i>see</i> the shortcut take over late, swapping <i>removes</i> it late, but retraining "
    "<i>fixes</i> it almost anywhere. Where a habit is found is not where it must be fixed. Caveat: full-block retraining on "
    "the same 10,000 reviews is a strong fix, which is why almost everything works.")

# ------------------------------------------------------------------ Part B
S.append(PageBreak())
h1("Part B. GPT-2 with LoRA (Sidhant)")
p("<b>Question (RQ3):</b> does the Part A picture hold when the model is fine-tuned with LoRA instead of full fine-tuning? "
  "We used two LoRA sizes, <b>rank 4</b> (tiny) and <b>rank 64</b> (larger), with LoRA on the attention and MLP weights of "
  "all 12 blocks. Every other setting (data, prompt, loss, batch, epochs, seeds, checks) is the same as Part A. All GPU "
  "runs used one GPU type (Colab NVIDIA L4) with pinned software versions. After training, the LoRA add-on is merged into "
  "the weights, so the resulting models are ordinary GPT-2 models and Part A’s tools run on them unchanged. "
  "Results: RESULTS_LORA.md. Pull requests #1–#6 and #8–#13.")
h3("B1. Choosing the learning rate (pilot, then an amendment)")
p("LoRA needs a higher learning rate than full fine-tuning, so we pre-registered a pilot: try 1e-4, 3e-4 and 1e-3 on "
  "seed 0, both ranks, shortcut and clean twin, and pick the smallest rate at which <i>both</i> ranks pass all checks. "
  "<b>No rate passed for both ranks</b> (rank 4 learned the shortcut only at 1e-3; rank 64 failed the task check at 1e-3). "
  "As pre-registered, we stopped and reported to Sonu. He decided one extra rate, <b>5e-4</b>, with a per-rank fallback "
  "written down in advance. At 5e-4 both ranks passed, so all LoRA models use 5e-4. This decision came after seeing pilot "
  "results, so it is stated openly; seeds 1 and 2 are the independent check.")
h3("B2. Main models (12): all pass their checks")
table([["", "Full fine-tuning", "LoRA rank 4", "LoRA rank 64"],
       ["Shortcut gap (seeds 0 / 1 / 2)", "0.646 / 0.626 / 0.541", "0.353 / 0.266 / 0.224", "0.800 / 0.801 / 0.879"],
       ["Clean twins", "gap ≈ 0", "gap within ±0.015", "gap within ±0.015"],
       ["Accuracy on normal reviews", "about 0.87", "0.867–0.872", "0.869–0.879"]],
      [5.2, 3.8, 3.8, 3.7])
p("Rank 4 learns a much <b>weaker</b> shortcut, rank 64 a <b>stronger</b> one than full fine-tuning. Keep this in mind "
  "below: rank 4’s numbers are noisier because its gap is small.")
h3("B3. Logit lens: the same “overthinking” at both ranks (reproduces)")
p("For both ranks the true sentiment is readable up to about layer 9 (calibrated accuracy on “opposite” 0.85 there), and "
  "the late blocks then flip it. Calibrated accuracy on “opposite” at the final layer, mean of 3 seeds: full fine-tuning "
  "0.283 (the per-seed 0.242 / 0.268 / 0.339 from Part A2), rank 4 0.586, rank 64 0.049. The "
  "pre-registered reading is “overthinking” for all three. <b>RQ1’s answer holds under LoRA.</b>")
h3("B4. Swaps: mostly the same map, but the control fails")
p("We applied every Part A swap rule (67 rules) and asked whether each gets the same PASS/FAIL as full fine-tuning. "
  "Rank 4 reproduces 57 of 67, rank 64 reproduces 61 of 67. The differences:")
bullets(["<b>The shortcut is more concentrated in the late blocks under LoRA.</b> “Blocks 9–11 are late-specific” failed for "
         "full fine-tuning but <b>passes</b> for both ranks. Swapping in the clean twin’s blocks 9–10–11 removes essentially all "
         "of it (0.96–1.01 of the gap). For rank 64, blocks 9–10–11 alone can even <i>create</i> the shortcut in a clean "
         "model, which full fine-tuning could not.",
         "<b>The control (c-ii) fails for the late blocks at both ranks.</b> Blocks from <i>another seed’s</i> shortcut model "
         "keep only 0.70–1.55 of the gap instead of at least 0.80 in every seed. So for LoRA models we <b>cannot</b> claim "
         "“real removal, not breakage”, and we report this as “does not reproduce”.",
         "Ranking of single blocks by the swap: rank 4 matches full fine-tuning (Spearman 0.981 / 0.937 / 0.862); rank 64 "
         "does not in seed 2 (0.088), because in that seed blocks 0–9 all remove almost nothing, so their order is noise."])
h3("B5. Why the control fails: different seeds use different directions (exploratory)")
p("A guess for the c-ii failure: each LoRA model stores the shortcut in its own seed-specific directions, so a block from "
  "another seed “does not fit”. Sonu approved a one-evening exploratory check (no rule, method written down before "
  "computing). We measured how much the weight changes of different seeds point in the same directions, against a "
  "random baseline. Result: full fine-tuning seeds overlap strongly (0.41–0.68 at the same size), LoRA seeds much less "
  "(rank 4: 0.05–0.17; rank 64: only about twice the random level), in every block. On the input side, rank 64 is at the "
  "random level, which fits a known property of LoRA: its A matrix barely moves from its random start (Zhu et al. 2024). "
  "This supports the explanation; it does not prove it. The general fact that LoRA updates are poorly aligned is known "
  "(KnOTS, ICLR 2025); what is ours is the consequence for interpretability: <b>swap-based controls should not be trusted "
  "on LoRA models without checking alignment first.</b>")
h3("B6. Retraining: “fix anywhere” holds for rank 64, not for rank 4")
bullets(["<b>Rank 64:</b> every single block removes the shortcut in all seeds (12 of 12), like full fine-tuning. Swap and "
         "retrain do <i>not</i> rank the blocks alike (block 0 retrains best in two seeds while the swap points to blocks 10–11).",
         "<b>Rank 4:</b> only 6 of 12 single blocks pass (blocks 1–5 and 11 fail, always because the gap stays above half, never "
         "because of an accuracy cost). Here location matters, and swap and retrain agree on the order. Many of these failures "
         "are borderline (within test-set noise); the clear ones are in seed 2.",
         "With a short training budget (rank 64 only, as pre-registered), late blocks are not faster: the rule fails (F/F/F), the "
         "same verdict as full fine-tuning. Block 0 is the fastest (0.715 of the gap removed at step 100), block 11 among the "
         "slowest, although the swaps point to blocks 10–11."])
figure(REPO / "figures/lora/lora_r4/retrain_map.png", 16.0, "Figure B1 (from the repo). Rank 4: fraction of the shortcut removed by swapping (blue) vs "
       "retraining (red), per block. Retraining blocks 1–4 and 11 stays below the dashed 50% line.")
h3("B7. Is rank 4 different because of its rank or its weak shortcut? (descriptive check)")
p("Rank 4’s models learned a weak shortcut and fit their training data less well. To separate the two, we re-ran the "
  "retraining on one older pilot model that is also rank 4 but learned a <b>stronger</b> shortcut (trained at learning rate "
  "1e-3, gap 0.583). Result: all 12 single blocks remove the shortcut, and blocks 1–5 behave like full fine-tuning and rank "
  "64, not like our main rank-4 models. This <i>suggests</i> rank 4’s pattern comes from its weak shortcut, not from the rank "
  "itself. One seed and a different learning rate, so it is a hint, not proof.")
h3("B8. Weak fix: does localization tell you where a cheap fix works best? (new experiment)")
p("Full-block retraining is so strong that almost every block works, so it cannot say <i>which</i> block to fix. We "
  "pre-registered a weaker fix: a tiny LoRA adapter (49,152 parameters, about 140 times fewer than a full block) on "
  "<b>one</b> block at a time, trained for a short budget (100 steps, set beforehand from a run with the adapter on all "
  "blocks). We applied it to all three kinds of shortcut models (full fine-tuning, rank 4, rank 64), every block, 3 seeds: "
  "120 runs. Verdicts, read at the pre-registered step:")
table([["Rule (all 3 seeds needed)", "Full fine-tuning", "LoRA rank 4", "LoRA rank 64"],
       ["1. The swap-localized block is in the top 3 of 12", "FAIL (P/P/F)", "FAIL (P/F/P)", "FAIL (F/F/F)"],
       ["1b. The best block is one of the top-3 swap blocks", "FAIL (P/P/F)", "FAIL (F/F/F)", "FAIL (P/F/P)"],
       ["2. Swap map and cheap-fix map rank blocks alike", "<b>PASS</b> (0.99 / 0.99 / 0.92)", "FAIL", "FAIL"],
       ["3. Block 0 is in the top 3", "FAIL (F/F/P)", "FAIL (F/F/P)", "FAIL (F/P/P)"],
       ["Control: fixing on shortcut data keeps the shortcut", "PASS", "PASS", "PASS"]],
      [6.2, 3.6, 3.3, 3.4])
bullets(["<b>Full fine-tuning:</b> the swap map ranks the blocks almost exactly like the cheap fix does; this is the "
         "strongest “find predicts fix” link in the project. But it does not reliably pick the single best block: the localized "
         "block 11 ranks 3rd, 2nd and 4th (seed 2 misses third place by 0.03).",
         "<b>LoRA:</b> the link breaks at both ranks. For rank 4 the cheap fix barely works within 100 steps (close to noise); for "
         "rank 64 the best blocks are 11, 0 and 11 while the localized block 10 ranks 4th–5th.",
         "The verdicts shift if read at other steps (for example rank 4’s Rule 1 passes at step 200); this is reported in full."])
box("<b>Takeaway of Part B.</b> Where the shortcut is <i>seen</i> does not change with LoRA. Where it is <i>stored</i> mostly "
    "holds and is even more concentrated in the late blocks, but the swap control fails. Where it can be <i>fixed</i>: rank 64 "
    "like full fine-tuning, rank 4 differently (probably because of its weak shortcut). Under full fine-tuning, the swap map "
    "is a good guide to where a cheap fix works; under LoRA it is not. Together with the failed control and the low seed "
    "overlap, this gives one consistent message: <b>swap-based localization is less trustworthy as a guide to fixing on LoRA "
    "models.</b>")
small("Caveats to keep in mind: LoRA uses a 10× higher learning rate than full fine-tuning (5e-4 vs 5e-5); rank 4’s shortcut "
      "is weak, so its numbers are noisier; retraining whole blocks of a merged LoRA model is a strong fix; one data set, one "
      "planted shortcut, GPT-2 small, 3 seeds.")

# ------------------------------------------------------------------ Part C
S.append(PageBreak())
h1("Part C. Wrong-label memorization (Siddhant)")
p("<b>Question:</b> a second, different kind of bad habit. If some training labels are wrong, does the model memorize them, "
  "and does that memorization live in the same place as the shortcut, and is it also fixable anywhere? This follows "
  "Dankers & Titov (2024), who studied memorization of wrong labels by swapping and retraining layers. "
  "Pull requests #7, #16–#19.")
h3("C1. Setup")
bullets(["<b>Data:</b> SST-2 (short movie-review phrases, positive/negative), deduplicated, at least 5 words, 6,920 training "
         "examples, balanced. In 15% of them (1,038 examples, called <b>F</b>) the label is flipped to the wrong one. The rest "
         "(called <b>C</b>) keep their true labels. Test set: the 872 SST-2 validation sentences with true labels.",
         "<b>Models:</b> GPT-2 small, full fine-tuning, the same prompt, label words, loss and settings as Part A. "
         "<b>Noisy model</b> = trained on the flipped labels; <b>clean twin</b> = same examples, true labels. 3 seeds each.",
         "<b>The habit</b> is measured as the <b>memorization rate M</b>: the share of the 1,038 flipped examples for which the "
         "model predicts the <i>wrong</i> (flipped) label it was taught."])
h3("C2. How long to train (pre-registered rule)")
p("Memorizing wrong labels takes many passes over the data, so the length was set by a rule: train the seed-0 noisy model "
  "until M ≥ 0.90, checked after each epoch, at most 30 epochs. It got there at <b>epoch 7</b> (M went 0.14, 0.24, 0.35, "
  "0.53, 0.71, 0.90, 0.93 over epochs 1–7). All 6 final models then trained for exactly 7 epochs.")
h3("C3. Results: all 6 models pass their checks")
table([["Model", "Memorization M", "Accuracy on test set", "Accuracy on C (true labels)", "Checks"],
       ["Noisy, seed 0 / 1 / 2", "0.926 / 0.955 / 0.920", "0.845 / 0.857 / 0.849", "0.993 / 0.991 / 0.990", "pass"],
       ["Clean, seed 0 / 1 / 2", "0.002 / 0.001 / 0.001", "0.916 / 0.911 / 0.911", "0.999 / 0.999 / 0.999", "pass"]],
      [3.6, 3.4, 3.4, 3.8, 2.3])
p("So the noisy models memorize more than 90% of the wrong labels while still getting the normal training examples right "
  "and staying above 0.80 on the test set; the clean twins do not memorize. For reference, GPT-2 before fine-tuning "
  "“agrees” with the wrong label on only 41% of F (below chance). An overlap count shows that 55% of the flipped phrases (566 "
  "of 1,038) share text with a correctly labelled phrase, so many memorized labels directly contradict related, correctly "
  "labelled text.")
h3("C4. What went wrong along the way (documented in results/wronglabel/RUN_NOTES.md)")
bullets(["A first set of final runs on free Colab (T4 GPU) was lost when the session ended before the files were saved; all 6 "
         "were redone from scratch. They were replaced because the files were lost, not because of their numbers.",
         "The final runs were therefore trained on Siddhant’s laptop GPU (GTX 1650), while the epoch probe ran on a Colab T4. "
         "The next phase (W2) stays on the laptop GPU.",
         "One run crashed with a GPU memory error and was restarted from step 0 with the same settings.",
         "Rebuilding the data on Colab gave different file hashes; the script stopped as designed, and the original files "
         "(which pass the hash check) were used instead. The cause was not found."])
h3("C5. Next: phase W2 (pre-registered and approved 27 Oct, not run yet)")
p("The same three tools as the shortcut study, on these 6 models: the <b>logit lens</b> (is the true label readable in the "
  "middle layers before the wrong label takes over?), <b>block swaps</b> with the same cases and controls, and <b>targeted "
  "retraining</b> of each block, with a gate run first. The written prediction is: “a memorized wrong label has no single "
  "cue to cancel, so retraining one block should not remove it”. It holds only if no single block passes the “removes "
  "memorization” rule in all three seeds. This is the opposite of what the shortcut showed, which makes it a clear test.")
p("One difference from Dankers &amp; Titov to keep in mind: our retraining uses all 6,920 examples, the flipped ones with "
  "their true labels, while they retrain only on the correctly labelled examples. Our fix is therefore stronger, which is "
  "one reason the W2 prediction may fail.")

# ------------------------------------------------------------------ Part D
h1("Part D. Pythia-160M (Piyush)")
p("<b>Question:</b> do the GPT-2 findings reproduce on a different model family? Pythia-160M is about the same size as "
  "GPT-2 small but was trained on different data and runs attention and MLP in parallel. We only report “reproduces” or “does "
  "not reproduce”. The plan is full fine-tuning first (with its own learning-rate pilot), then the same tools, then LoRA "
  "with the settings from Part B. Pull requests #14, #15 (merged) and #20 (open). <b>No training has started yet</b>, "
  "because the pre-run checks found problems with our data on this model.")
h3("D1. What the pre-run checks found")
bullets(["The label words “ positive” and “ negative” are single tokens in Pythia’s tokenizer: fine.",
         "<b>The names do not split into equal pieces:</b> name B (“Dorvan Hastrow”) is 5 tokens in Pythia while A is 4. The "
         "checks require equal token counts.",
         "<b>Untrained Pythia already prefers one name:</b> the pre-registered rule allows a difference in average score "
         "between A and B of at most 10% of the score spread. Measured: 0.092 against an allowed 0.047, so it fails."])
h3("D2. Precision and prompt: checked twice, as the process intends")
bullets(["The first name check accidentally loaded the model in half precision (fp16). Piyush kept that result for the record, "
         "pre-registered an fp32 rule, and re-ran: still a fail.",
         "In review, Sonu noticed that the committed check code did not use the project’s prompt (“Review: …” followed by "
         "“Sentiment:”). The open pull request #20 rewrites the check to use the shared prompt code and logs the prompt. Its "
         "result (0.091603, run on a Mac) differs from the earlier file (0.091589, run on an NVIDIA GPU) only in the 5th "
         "decimal, which a prompt change could not produce, so the earlier run must already have used the prompt; the tiny "
         "difference is hardware noise. Either way, <b>the original names clearly fail</b>."])
h3("D3. Where it stands, and the fix (from the open PR #20, not merged yet)")
p("Piyush had pre-registered a fixed list of 20 invented replacement names and a strict rule: keep only names that are "
  "exactly 4 tokens (the length of name A), then take the first pair that passes the preference check. According to PR "
  "#20’s token-count file, <b>only 1 of the 20 candidates is 4 tokens</b>, so under that rule no pair can be formed. "
  "However, 15 of the 20 candidates are 5 tokens. <b>The fix</b> (Sonu’s amendment): A and B only need the <i>same</i> "
  "token count as each other. The first eligible pair is then #2 / #3 (Morveth Dask / Talvick Renlo), if it passes the "
  "10% preference check. This change goes into the spec before any Pythia training. Because Pythia will then use "
  "different names than GPT-2, that difference will be stated whenever the two models are compared.")

# ------------------------------------------------------------------ 9 overall
S.append(PageBreak())
h1("6. Where the whole project stands")
table([["Track", "Done", "Next", "Main open issue"],
       ["A. GPT-2 full FT (Sonu)", "Everything: models, lens, swaps, retraining, budget.", "Report frame, GPT-2 results, answers to the TA.", "None."],
       ["B. LoRA (Sidhant)", "All experiments, incl. the weak fix and two follow-up checks. 12 PRs merged.", "RESULTS_LORA write-up, plain-words report, LoRA sections of the paper.", "None; no new experiments unless agreed."],
       ["C. Wrong labels (Siddhant)", "Data, 6 models, all checks pass. W2 plan approved.", "Run W2: lens, swaps, retraining.", "Time: W2 is a full set of experiments."],
       ["D. Pythia (Piyush)", "Pre-run checks; precision and prompt double-checked.", "Name-rule amendment into the spec, new data, lr pilot, 6 models, then the tools and LoRA.", "Most of the track is still ahead."]],
      [3.4, 4.6, 4.3, 4.2])
p("<b>Timeline.</b> The team plan had main runs, logit lens and swaps for each track by 7 November, retraining by 14 "
  "November, and an <b>experiment freeze on 21 November</b>, followed by the report and slides (early December). Tracks A and "
  "B are finished ahead of this. Tracks C and D are behind the original plan, D the most. <b>Cut plan:</b> if time runs "
  "short, we cut Pythia LoRA and Pythia retraining first, never the checks or the controls.")
h3("What we can already say")
p("<b>Central hypothesis:</b> where a fine-tuned model shows a bad habit is not where it is easiest to fix, and this "
  "link is weaker under LoRA. The results so far support it:")
bullets(["<b>RQ1 (where it takes over):</b> late, in the last blocks (9–11), for GPT-2 with full fine-tuning and with LoRA at "
         "both ranks.",
         "<b>RQ2 (is that the best place to fix it?):</b> not the only place. With strong retraining almost any block fixes it. "
         "With a cheap fix, the swap map points to the right <i>area</i> under full fine-tuning (it ranks the blocks well) but "
         "does not reliably pick the single best block.",
         "<b>RQ3 (LoRA):</b> seeing the shortcut is unchanged; where it is stored mostly holds; but the swap control fails, seeds "
         "learn different directions, and the swap map stops being a guide to cheap fixes. RQ3 across model families "
         "(Pythia) and the second habit (wrong labels) are still open."])
h3("Limitations (to state honestly in the final report)")
bullets(["One planted shortcut on one data set, GPT-2 small, 3 seeds per condition.",
         "Full-block retraining on the same reviews the model was trained on is a strong fix.",
         "LoRA and full fine-tuning use different learning rates, and rank 4’s shortcut is weak (noisy numbers).",
         "Our setup is a simplified version of Eshuijs et al.’s (our own names, logit lens and block swaps instead of their "
         "path patching), and of Dankers & Titov’s (GPT-2 small, prompt-based labels, 7 epochs instead of 50)."])
h3("Next steps")
table([["Who", "Next"],
       ["Sonu", "Review and merge PRs (Pythia #20 and the name decision); report frame, GPT-2 results, TA answers."],
       ["Sidhant", "RESULTS_LORA write-up and a 1–2 page plain-words LoRA report; draft the LoRA method and results sections of the paper; be the contact for the LoRA pipeline when Pythia reaches its LoRA step."],
       ["Siddhant", "Run phase W2 (lens, swaps, retraining) on the laptop GPU; report and write-up."],
       ["Piyush", "Name-rule amendment into the spec, then data_pythia/, learning-rate pilot, 6 models, the tools, then Pythia LoRA."]],
      [2.4, 14.1])

h3("Questions we expect at the review, and our answers")
table([["Question", "Answer"],
       ["What is your central hypothesis?", "Where a fine-tuned model shows a bad habit is not where it is easiest to fix, and this link is weaker under LoRA. The three research questions test it piece by piece, and the results so far support it."],
       ["Why the logit lens, and not activation patching?", "The lens only tells us where the answer becomes readable. The causal tests are the block swaps (with controls) and the retraining. That is our response to the TA’s point that patching results depend on the method."],
       ["How will you fix the Pythia name problem?", "Amend the rule so names A and B only need the same token count as each other. 15 of the 20 pre-registered candidates are 5 tokens, so the first pair is #2 / #3 (Morveth Dask / Talvick Renlo), if it passes the 10% preference check. The amendment goes into the spec before any training."],
       ["Will tracks C and D finish by the 21 November freeze?", "They are behind (reasons in C4 and D2). If time runs short, we cut Pythia LoRA and Pythia retraining first, never the checks or the controls."],
       ["How does LoRA work? What do A and B do?", "The original weights stay frozen. Next to each weight matrix we train two small ones: A squeezes the 768 input numbers down to r numbers, B expands them back to the output size. Their product B·A (times alpha/r = 2) is the change added to the weight. B starts at zero, so training starts from the unchanged model. After training we merge it in: W + 2·B·A (Section 5.3)."],
       ["What do rank 4 and rank 64 mean?", "r is how many numbers the change is squeezed through, i.e. how many independent directions it can use. Rank 4 trains 589,824 numbers (0.47% of GPT-2), rank 64 trains 9,437,184 (7.58%). Full fine-tuning changes all 124 million."],
       ["Isn’t a 10× higher learning rate for LoRA unfair?", "LoRA trains far fewer numbers and starts from a zero change, so it needs a larger step; this is standard. We chose it by a pre-registered pilot plus a written amendment (B1), and list it as a limitation."]],
      [5.0, 11.5])

# ------------------------------------------------------------------ appendix
h1("Appendix: where things are in the repository")
table([["File or folder", "What it is"],
       ["TEAM_START.md", "Start here: the project in one page, the rules, who does what, the timeline."],
       ["PROJECT_SPEC.md", "Every pre-registered plan and rule, in order (Pilot, Phases 2–3b, W1–W2, L1–L3b, P1)."],
       ["SUMMARY.md, RESULTS.md", "GPT-2 full fine-tuning: one-page summary and all tables."],
       ["RESULTS_LORA.md", "Everything from the LoRA track, generated from results files."],
       ["results/wronglabel/RUN_NOTES.md", "Wrong-label runs, every attempt and what went wrong."],
       ["results/", "One JSON file per seed for every experiment."],
       ["src/", "All code: data, training, checks, logit lens, swaps, retraining, drivers, report scripts."],
       ["HANDOFF_LORA.md, HANDOFF_WRONGLABEL.md", "The original instructions for the LoRA, Pythia and wrong-label tracks."]],
      [5.4, 11.1])
small("Papers mentioned: Halawi et al. 2024 (Overthinking the Truth); Eshuijs et al. 2025 (Short-circuiting Shortcuts); "
      "Hase et al. 2023 (Does Localization Inform Editing?); Dankers & Titov 2024 (Generalisation First, Memorisation "
      "Second?); Hu et al. 2021 (LoRA); Zhu et al. 2024 (Asymmetry in Low-Rank Adapters); Stoica et al. 2025 (KnOTS).")


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DV", 7.5)
    canvas.setFillColor(MUTED)
    if doc.page > 1:
        canvas.drawString(2 * cm, 1.2 * cm, "Find vs Fix: progress report for Review 1 (28 Oct 2026)")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"page {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2 * cm,
                        bottomMargin=2 * cm, title="Find vs Fix: progress report for Review 1",
                        author="Team Attention is All we need")
doc.build(S, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
