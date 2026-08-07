"""surya-ocr-2 task prompts — sourced VERBATIM from datalab-to/surya's
surya/inference/prompts.py and surya/settings.py (2026-07-30). "The exact wording is the
model's training-time contract — do not paraphrase without retraining" (upstream's own comment).

Do NOT invent prompts for this model — an imprecise paraphrase (e.g. "Read the text in this
image.") can still produce plausible-looking output, which is exactly what makes it a trap: it
looked correct during ad-hoc testing here until checked against the real source.
"""

# Bbox coordinates in all tasks are "x0 y0 x1 y1" strings, normalized 0-1000 (BBOX_SCALE).
BBOX_SCALE = 1000

# A single page can need 2500-7000+ INPUT tokens (vision patches) depending on resolution;
# pad every task's max_length budget by this much so --max_length isn't the thing that fails.
IMAGE_TOKEN_HEADROOM = 8000

LAYOUT_PROMPT = (
    'Output the layout of this image as JSON. Each entry is a dict with '
    '"label", "bbox", and "count" fields. Bbox is x0 y0 x1 y1, normalized 0-1000.'
)

BLOCK_PROMPT = "OCR this block image to HTML."  # per-block OCR on a CROPPED region, not a full page

TABLE_REC_PROMPT = (
    'Output the table rows then columns as JSON. Each entry is a dict with '
    '"label" ("Row" or "Col") and "bbox" (x0 y0 x1 y1, normalized 0-1000).'
)

HIGH_ACCURACY_BBOX_PROMPT = (  # full-page OCR to HTML — what most people mean by "just OCR this"
    "OCR this image to HTML. Each block is a div with data-label and data-bbox "
    "(x0 y0 x1 y1, normalized 0-1000)."
)

# task key -> (prompt, default max_tokens). max_tokens mirror surya/settings.py's
# SURYA_MAX_TOKENS_{LAYOUT,TABLE_REC,FULL_PAGE}. "block" is excluded from PROMPT_MAPPING
# (below) for full-page eval since it needs a prior layout pass + per-block image crops, not a
# single full-page request; it's kept here for completeness / future use.
TASKS = {
    "layout": (LAYOUT_PROMPT, 3072),
    "table_rec": (TABLE_REC_PROMPT, 3072),
    "ocr": (HIGH_ACCURACY_BBOX_PROMPT, 12288),  # = surya's "high_accuracy_bbox" full-page prompt
    "block": (BLOCK_PROMPT, 8192),              # needs a cropped block image, not a full page
}

# Full-page tasks only (excludes "block") — what eval.py runs by default.
FULL_PAGE_TASKS = {k: v for k, v in TASKS.items() if k != "block"}

LAYOUT_LABEL_SET = {
    "Caption", "Footnote", "Equation-Block", "List-Group", "Page-Header", "Page-Footer",
    "Image", "Section-Header", "Table", "Text", "Complex-Block", "Code-Block", "Form",
    "Table-Of-Contents", "Figure", "Chemical-Block", "Diagram", "Bibliography", "Blank-Page",
}
TABLE_REC_LABEL_SET = {"Row", "Col"}
