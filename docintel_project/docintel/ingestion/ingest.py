"""Module 1: PDF ingestion and parsing -> Chunk objects / data/chunks.jsonl.

Run from the directory that contains `data/` (e.g. /home/claude):
    python -m docintel.ingestion.ingest
"""
import base64
import hashlib
import json
import logging
import os
import re
import statistics
from typing import List, Optional


import pymupdf as fitz

from docintel.schemas import Chunk, DOCS_DIR, PAGES_DIR, CHUNKS_PATH, DATA_DIR

log = logging.getLogger("docintel.ingestion")

CACHE_DIR = os.path.join(DATA_DIR, "cache")
RENDER_DPI = 150
CHUNK_CHARS = 800
MIN_PAGE_TEXT = 50
VLM_MODEL_DEFAULT = "claude-sonnet-4-5"

VLM_PROMPT = (
    "You are analysing a chart or figure from a business document. Describe it for a search index. "
    "Provide: 1) chart type, 2) title, 3) axes with units, 4) EVERY visible data point/value "
    "(series name, category, value) as a list, 5) the key trend in one sentence. "
    "Be exact with numbers; do not guess values that are not visible."
)
VLM_READ_PROMPT = "Transcribe all text on this page exactly, preserving reading order. Render tables as markdown."


# --------------------------------------------------------------------------- VLM / cache
def _model() -> str:
    return os.environ.get("DOCINTEL_VLM_MODEL", VLM_MODEL_DEFAULT)


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _cache_get(key: str) -> Optional[str]:
    try:
        with open(os.path.join(CACHE_DIR, key + ".json"), encoding="utf-8") as f:
            return json.load(f)["text"]
    except Exception:
        return None


def _cache_put(key: str, text: str):
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(os.path.join(CACHE_DIR, key + ".json"), "w", encoding="utf-8") as f:
            json.dump({"text": text}, f)
    except Exception as e:  # cache is best effort
        log.warning("could not write cache: %s", e)


def _vlm_call(image_path: str, prompt: str, max_tokens: int = 1500) -> str:
    import anthropic
    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode()
    ext = os.path.splitext(image_path)[1].lower()
    media = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=_model(), max_tokens=max_tokens,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}},
            {"type": "text", "text": prompt}]}])
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()


def describe_visual(image_path: str, page: Optional[int] = None) -> str:
    """Describe a chart/figure image with a Claude vision call (cached by image hash).

    Without ANTHROPIC_API_KEY (or on API failure) returns a placeholder so the pipeline still runs.
    """
    placeholder = (f"[chart on page {page if page is not None else '?'} - "
                   "visual description unavailable, analyze page image]")
    key = "vis_" + hashlib.sha256((_file_hash(image_path) + _model() + VLM_PROMPT).encode()).hexdigest()
    cached = _cache_get(key)
    if cached:
        return cached
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return placeholder
    try:
        text = _vlm_call(image_path, VLM_PROMPT)
    except Exception as e:
        log.warning("describe_visual failed for %s: %s", image_path, e)
        return placeholder
    if text:
        _cache_put(key, text)
        return text
    return placeholder


def vlm_read_page(image_path: str) -> str:
    """Optional: transcribe a page image with Claude vision. Returns "" if no API key / failure."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return ""
    key = "read_" + hashlib.sha256((_file_hash(image_path) + _model() + VLM_READ_PROMPT).encode()).hexdigest()
    cached = _cache_get(key)
    if cached is not None:
        return cached
    try:
        text = _vlm_call(image_path, VLM_READ_PROMPT, max_tokens=3000)
    except Exception as e:
        log.warning("vlm_read_page failed for %s: %s", image_path, e)
        return ""
    if text:
        _cache_put(key, text)
    return text


# --------------------------------------------------------------------------- OCR
def _preprocess_for_ocr(img):
    """grayscale -> autocontrast -> deskew (projection profile). Returns PIL image."""
    import numpy as np
    from PIL import Image, ImageOps
    g = ImageOps.autocontrast(img.convert("L"), cutoff=1)
    small = g.copy()
    small.thumbnail((900, 900))
    arr = (np.asarray(small) < 128).astype(np.float32)
    best_angle, best_score = 0.0, -1.0
    if arr.sum() > 50:
        for a in np.arange(-6, 6.01, 0.5):
            r = np.asarray(Image.fromarray((arr * 255).astype("uint8")).rotate(
                a, resample=Image.BILINEAR, fillcolor=0)).astype(np.float32)
            score = float(np.var(r.sum(axis=1)))
            if score > best_score:
                best_angle, best_score = float(a), score
    if abs(best_angle) >= 0.5:
        g = g.rotate(best_angle, resample=Image.BICUBIC, expand=True, fillcolor=255)
    return g


def ocr_image(image_path: str) -> str:
    """OCR an image with pytesseract. Returns "" (and logs) if tesseract is unavailable."""
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(image_path)
        # fix 90-degree orientation if tesseract OSD can tell
        try:
            osd = pytesseract.image_to_osd(img.convert("L"))
            m = re.search(r"Rotate: (\d+)", osd)
            if m and int(m.group(1)) in (90, 180, 270):
                img = img.rotate(-int(m.group(1)), expand=True, fillcolor="white")
        except Exception:
            pass
        proc = _preprocess_for_ocr(img)
        return pytesseract.image_to_string(proc, config="--psm 6").strip()
    except pytesseract_missing_errors() as e:
        log.warning("Tesseract OCR unavailable (%s); install tesseract-ocr to OCR scanned pages.", e)
        return ""
    except Exception as e:
        log.warning("OCR failed for %s: %s", image_path, e)
        return ""


def pytesseract_missing_errors():
    try:
        import pytesseract
        return (pytesseract.TesseractNotFoundError, ImportError)
    except ImportError:
        return (ImportError,)


# --------------------------------------------------------------------------- text helpers
def _norm(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s.replace(" ", " ")).strip()


def _split_text(text: str, limit: int = CHUNK_CHARS) -> List[str]:
    """Split on sentence boundaries into pieces of at most ~limit chars."""
    if len(text) <= limit:
        return [text]
    sents = re.split(r"(?<=[.!?])\s+", text)
    out, cur = [], ""
    for s in sents:
        while len(s) > limit:  # pathological: no sentence breaks
            cut = s.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            if cur:
                out.append(cur)
                cur = ""
            out.append(s[:cut])
            s = s[cut:].strip()
        if cur and len(cur) + 1 + len(s) > limit:
            out.append(cur)
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        out.append(cur)
    return out


def _union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _rnd(b):
    return [round(float(x), 1) for x in b]


def _inside(inner, outer, tol=2.0):
    cx, cy = (inner[0] + inner[2]) / 2, (inner[1] + inner[3]) / 2
    return outer[0] - tol <= cx <= outer[2] + tol and outer[1] - tol <= cy <= outer[3] + tol


def _page_blocks(page):
    """Return text blocks: dict(bbox, text, size, bold, nlines)."""
    blocks = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        lines, sizes, bold_chars, chars = [], [], 0, 0
        for ln in b["lines"]:
            txt = "".join(sp["text"] for sp in ln["spans"])
            if not txt.strip():
                continue
            lines.append(txt.strip())
            for sp in ln["spans"]:
                n = len(sp["text"].strip())
                if n:
                    sizes.append((sp["size"], n))
                    chars += n
                    if (sp["flags"] & 16) or "bold" in sp["font"].lower() or "black" in sp["font"].lower():
                        bold_chars += n
        if not lines:
            continue
        tot = sum(n for _, n in sizes)
        size = sum(s * n for s, n in sizes) / tot
        text = _norm(" ".join(lines))
        blocks.append({"bbox": list(b["bbox"]), "text": text, "size": size,
                       "bold": chars > 0 and bold_chars / chars > 0.6, "nlines": len(lines)})
    return blocks


def _order_blocks(blocks, page_width):
    """Reading order: blocks crossing the page midline split the page into bands; within a band
    go column by column (left then right), top to bottom."""
    if not blocks:
        return []
    blocks = sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))
    mid = page_width / 2
    crosses = lambda b: b["bbox"][0] < mid - 5 and b["bbox"][2] > mid + 5
    out, band = [], []

    def flush():
        if not band:
            return
        left = [b for b in band if b["bbox"][2] <= mid + 5]
        right = [b for b in band if b["bbox"][0] >= mid - 5]
        if left and right and len(left) + len(right) == len(band):
            out.extend(sorted(left, key=lambda b: b["bbox"][1]))
            out.extend(sorted(right, key=lambda b: b["bbox"][1]))
        else:
            out.extend(sorted(band, key=lambda b: (round(b["bbox"][1]), b["bbox"][0])))
        band.clear()

    for b in blocks:
        if crosses(b):
            flush()
            out.append(b)
        else:
            band.append(b)
    flush()
    return out


def _is_heading(b, body_size):
    t = b["text"]
    if len(t) > 100 or len(t) < 2 or b["nlines"] > 2:
        return False
    if re.match(r"^(figure|fig\.|table)\s*\d+", t, re.I):
        return False
    if b["size"] >= body_size * 1.15:
        return True
    return b["bold"] and b["size"] >= body_size * 0.98 and not t.endswith((".", ",", ";", ":"))


# --------------------------------------------------------------------------- tables
def _md_cell(c):
    return _norm(str(c if c is not None else "")).replace("\n", " ").replace("|", "\\|")


def table_to_markdown(rows) -> str:
    rows = [[_md_cell(c) for c in r] for r in rows if r and any(c not in (None, "") for c in r)]
    if len(rows) < 1:
        return ""
    n = max(len(r) for r in rows)
    rows = [r + [""] * (n - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "|".join(["---"] * n) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def _extract_tables(plumber_page):
    out = []
    try:
        found = plumber_page.find_tables()
    except Exception as e:
        log.warning("pdfplumber table detection failed: %s", e)
        return out
    for t in found:
        try:
            rows = t.extract()
        except Exception:
            continue
        if not rows or len(rows) < 2 or max(len(r) for r in rows) < 2:
            continue
        md = table_to_markdown(rows)
        if md:
            out.append((md, list(t.bbox)))
    return out


# --------------------------------------------------------------------------- figures
def _merge_rects(rects, margin=25.0):
    rects = [fitz.Rect(r) for r in rects]
    changed = True
    while changed:
        changed = False
        out = []
        for r in rects:
            for i, o in enumerate(out):
                grown = fitz.Rect(o.x0 - margin, o.y0 - margin, o.x1 + margin, o.y1 + margin)
                if grown.intersects(r):
                    out[i] = o | r
                    changed = True
                    break
            else:
                out.append(r)
        rects = out
    return rects


def _detect_figures(page, table_bboxes):
    """Return list of fitz.Rect for embedded images and dense vector-drawing regions."""
    pr = page.rect
    page_area = pr.width * pr.height
    figs = []
    try:
        for info in page.get_image_info():
            r = fitz.Rect(info["bbox"]) & pr
            if r.is_empty:
                continue
            if r.width * r.height > 0.85 * page_area:   # full-page scan, not a figure
                continue
            if r.width < 60 or r.height < 40:
                continue
            figs.append(r)
    except Exception as e:
        log.warning("image detection failed: %s", e)
    try:
        rects, count = [], 0
        for d in page.get_drawings():
            r = fitz.Rect(d["rect"])
            if r.is_empty and (r.width < 0.01 and r.height < 0.01):
                continue
            if any(_inside([r.x0, r.y0, r.x1, r.y1], tb) for tb in table_bboxes):
                continue
            if r.width * r.height > 0.6 * page_area:
                continue
            if (r.width > 0.7 * pr.width and r.height < 3) or (r.height > 0.7 * pr.height and r.width < 3):
                continue   # page rules / borders
            rects.append(r)
        merged = _merge_rects(rects)
        for m in merged:
            n = sum(1 for r in rects if m.contains(r + (-0.5, -0.5, 0.5, 0.5)) or m.intersects(r))
            if m.width >= 100 and m.height >= 80 and n >= 5:
                figs.append(m)
    except Exception as e:
        log.warning("vector drawing detection failed: %s", e)
    # dedupe overlapping (image inside a vector region etc.)
    figs = _merge_rects(figs, margin=0)
    return sorted(figs, key=lambda r: (r.y0, r.x0))


def _near_text(rect, blocks, exclude_boxes=(), body_size=10.0, margin=40):
    """Return (caption, other_text) of blocks inside or adjacent to the figure rect."""
    caption, other = [], []
    for b in blocks:
        bb = b["bbox"]
        if any(_inside(bb, eb) for eb in exclude_boxes):
            continue
        overlap_x = min(bb[2], rect.x1) - max(bb[0], rect.x0) > 0
        inside = _inside(bb, [rect.x0, rect.y0, rect.x1, rect.y1])
        near = overlap_x and (0 <= bb[1] - rect.y1 <= margin or 0 <= rect.y0 - bb[3] <= margin)
        if not (inside or near):
            continue
        if re.match(r"^(figure|fig\.|chart)\s*\d+", b["text"], re.I):
            caption.append(b["text"])
        elif _is_heading(b, body_size) and not inside:
            continue
        elif inside or len(b["text"]) < 300:
            other.append(b["text"])
    return " ".join(caption), " ".join(other)


# --------------------------------------------------------------------------- main per-PDF
def ingest_pdf(pdf_path: str, out_dir: str = PAGES_DIR) -> List[Chunk]:
    import pdfplumber
    os.makedirs(out_dir, exist_ok=True)
    doc_id = os.path.basename(pdf_path)
    stem = os.path.splitext(doc_id)[0]
    chunks: List[Chunk] = []
    section = ""
    fdoc = fitz.open(pdf_path)
    try:
        plumber = pdfplumber.open(pdf_path)
    except Exception as e:
        log.warning("pdfplumber could not open %s: %s", pdf_path, e)
        plumber = None
    zoom = RENDER_DPI / 72.0
    try:
        for pno, page in enumerate(fdoc, start=1):
            img_path = os.path.join(out_dir, f"{stem}_p{pno}.png")
            page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False).save(img_path)
            mk = lambda cid, typ, content, bbox=None: Chunk(
                chunk_id=cid, doc_id=doc_id, page=pno, section=section, type=typ,
                content=content, page_image=img_path, bbox=_rnd(bbox) if bbox else None)

            blocks = _page_blocks(page)
            total_chars = sum(len(b["text"]) for b in blocks)

            # ---- scanned / low text -> OCR
            if total_chars < MIN_PAGE_TEXT:
                text = ocr_image(img_path)
                if len(text) < MIN_PAGE_TEXT:
                    vlm = vlm_read_page(img_path)
                    text = vlm if len(vlm) > len(text) else text
                if text:
                    first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
                    page_section = section or first[:100]
                    c = mk(f"{stem}_p{pno}_ocr", "ocr", text, [0, 0, page.rect.width, page.rect.height])
                    c.section = page_section
                    chunks.append(c)
                else:
                    log.warning("%s p%d has no text layer and OCR/VLM produced nothing", doc_id, pno)
                continue

            # ---- tables
            tables = _extract_tables(plumber.pages[pno - 1]) if plumber else []
            tb_boxes = [bb for _, bb in tables]
            # ---- figures
            figs = _detect_figures(page, tb_boxes)

            # ---- text (reading order, section tracking)
            sizes = [(b["size"], len(b["text"])) for b in blocks]
            body_size = _weighted_median(sizes)
            ordered = _order_blocks(blocks, page.rect.width)
            buf, buf_boxes, t_n = [], [], 0

            def flush_text():
                nonlocal t_n
                if not buf:
                    return
                text = " ".join(buf)
                bbox = _union(buf_boxes)
                for piece in _split_text(text):
                    chunks.append(mk(f"{stem}_p{pno}_t{t_n}", "text", piece, bbox))
                    t_n += 1
                buf.clear()
                buf_boxes.clear()

            for b in ordered:
                if any(_inside(b["bbox"], tb) for tb in tb_boxes):
                    continue
                if any(_inside(b["bbox"], [f.x0, f.y0, f.x1, f.y1]) for f in figs):
                    continue
                if _is_heading(b, body_size):
                    flush_text()
                    section = b["text"]
                    continue
                if sum(len(x) for x in buf) + len(b["text"]) > CHUNK_CHARS and buf:
                    flush_text()
                buf.append(b["text"])
                buf_boxes.append(b["bbox"])
            flush_text()

            # ---- table chunks (section = nearest heading above, else current)
            heads = [(b["bbox"], b["text"]) for b in ordered if _is_heading(b, body_size)]

            def heading_above(bb):
                cand = [(y[1], t) for y, t in heads
                        if y[1] <= bb[1] + 2 and min(y[2], bb[2]) - max(y[0], bb[0]) > 0]
                return max(cand)[1] if cand else None
            for n, (md, bb) in enumerate(tables):
                above = heading_above(bb)
                c = mk(f"{stem}_p{pno}_tb{n}", "table", md, bb)
                if above:
                    c.section = above
                chunks.append(c)

            # ---- figures / charts
            for n, r in enumerate(figs):
                fig_path = os.path.join(out_dir, f"{stem}_p{pno}_fig{n}.png")
                try:
                    pad = fitz.Rect(r.x0 - 4, r.y0 - 4, r.x1 + 4, r.y1 + 4) & page.rect
                    page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=pad, alpha=False).save(fig_path)
                except Exception as e:
                    log.warning("could not crop figure %d on %s p%d: %s", n, doc_id, pno, e)
                    continue
                caption, other = _near_text(r, blocks, tb_boxes, body_size)
                visual = describe_visual(fig_path, page=pno)
                parts = []
                parts.append(f"Caption: {caption}" if caption else f"Figure {n + 1} on page {pno}")
                if other:
                    parts.append(f"Text near figure: {other}")
                parts.append(f"Visual description: {visual}")
                above = heading_above([r.x0, r.y0, r.x1, r.y1])
                c = mk(f"{stem}_p{pno}_c{n}", "chart", "\n".join(parts), [r.x0, r.y0, r.x1, r.y1])
                if above:
                    c.section = above
                chunks.append(c)
    finally:
        fdoc.close()
        if plumber:
            plumber.close()
    return chunks


def _weighted_median(pairs):
    if not pairs:
        return 10.0
    pairs = sorted(pairs)
    total = sum(n for _, n in pairs)
    acc = 0
    for s, n in pairs:
        acc += n
        if acc >= total / 2:
            return s
    return pairs[-1][0]


def ingest_folder(docs_dir: str = DOCS_DIR, chunks_path: str = CHUNKS_PATH) -> List[Chunk]:
    files = sorted(f for f in os.listdir(docs_dir) if f.lower().endswith(".pdf"))
    all_chunks: List[Chunk] = []
    for f in files:
        try:
            cs = ingest_pdf(os.path.join(docs_dir, f))
        except Exception as e:
            log.error("failed to ingest %s: %s", f, e, exc_info=True)
            continue
        log.info("%s: %d chunks", f, len(cs))
        all_chunks.extend(cs)
    os.makedirs(os.path.dirname(chunks_path) or ".", exist_ok=True)
    with open(chunks_path, "w", encoding="utf-8") as fh:
        for c in all_chunks:
            fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    counts = {}
    for c in all_chunks:
        counts[c.type] = counts.get(c.type, 0) + 1
    log.info("wrote %d chunks to %s: %s", len(all_chunks), chunks_path, counts)
    return all_chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ingest_folder()
