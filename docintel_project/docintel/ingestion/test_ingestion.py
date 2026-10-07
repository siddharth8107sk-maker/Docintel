"""Run from /home/claude:  python -m pytest docintel/ingestion/test_ingestion.py"""
import os
import shutil
import pytest
import pymupdf as fitz

from docintel.schemas import Chunk
from docintel.ingestion import ingest as ing

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.fixture(scope="session")
def chunks(tmp_path_factory):
    old = os.getcwd()
    os.chdir(ROOT)  # relative data/ paths resolve to /home/claude/data
    try:
        from docintel import make_sample_docs
        docs = tmp_path_factory.mktemp("docs")
        make_sample_docs.main(str(docs))
        out = tmp_path_factory.mktemp("pages")
        res = []
        for f in sorted(os.listdir(docs)):
            if f.endswith(".pdf"):
                res += ing.ingest_pdf(str(docs / f), str(out))
        yield res
    finally:
        os.chdir(old)


def test_required_fields(chunks):
    assert chunks
    ids = set()
    for c in chunks:
        assert isinstance(c, Chunk)
        assert c.chunk_id and c.doc_id.endswith(".pdf") and c.content.strip()
        assert c.type in ("text", "table", "chart", "image", "ocr")
        assert isinstance(c.section, str)
        assert c.chunk_id not in ids
        ids.add(c.chunk_id)
        d = c.to_dict()
        assert set(d) == {"chunk_id", "doc_id", "page", "section", "type", "content", "page_image", "bbox"}


def test_pages_one_based_and_images_exist(chunks):
    for c in chunks:
        assert c.page >= 1
        assert os.path.exists(c.page_image) and c.page_image.endswith(f"_p{c.page}.png")
        assert c.chunk_id.startswith(os.path.splitext(c.doc_id)[0] + f"_p{c.page}_")


def test_text_chunks_size_and_sections(chunks):
    text = [c for c in chunks if c.type == "text"]
    assert text and all(len(c.content) <= 1000 for c in text)
    assert any(c.section.startswith("4. Analysis") for c in text)


def test_tables(chunks):
    tbs = [c for c in chunks if c.type == "table"]
    assert {c.doc_id for c in tbs} >= {"factory_report_2025.pdf", "maintenance_summary_2025.pdf"}
    t = next(c for c in tbs if c.doc_id == "factory_report_2025.pdf")
    lines = t.content.splitlines()
    assert lines[0].startswith("| Quarter") and set(lines[1]) <= set("|-")
    assert "| Q2 | 92 |" in t.content


def test_two_column_reading_order(chunks):
    m = [c for c in chunks if c.doc_id == "maintenance_summary_2025.pdf" and c.type == "text"]
    secs = [c.section for c in m]
    assert secs.index("1. Scope") < secs.index("3. Findings") < secs.index("4. Actions Taken")


def test_charts(chunks):
    ch = [c for c in chunks if c.type == "chart"]
    assert len(ch) >= 2
    assert any("Figure 1" in c.content for c in ch)
    for c in ch:
        assert "Visual description" in c.content and c.bbox


def test_scanned_page_gets_ocr(chunks):
    ocr = [c for c in chunks if c.type == "ocr"]
    if shutil.which("tesseract") is None:
        pytest.skip("tesseract not installed")
    assert len(ocr) == 1 and ocr[0].doc_id == "scanned_quality_audit.pdf"
    assert ocr[0].chunk_id == "scanned_quality_audit_p1_ocr"
    low = ocr[0].content.lower()
    assert "defect rate" in low and "3.0%" in low


def test_describe_visual_placeholder_without_key(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(ing, "CACHE_DIR", str(tmp_path / "cache"))
    from PIL import Image
    p = tmp_path / "x.png"
    Image.new("RGB", (10, 10), "white").save(p)
    assert ing.describe_visual(str(p), page=7) == "[chart on page 7 - visual description unavailable, analyze page image]"


def test_describe_visual_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setattr(ing, "CACHE_DIR", str(tmp_path / "cache"))
    calls = []
    monkeypatch.setattr(ing, "_vlm_call", lambda path, prompt, max_tokens=0: calls.append(1) or "bar chart: Q1=1")
    from PIL import Image
    p = tmp_path / "y.png"
    Image.new("RGB", (10, 10), "red").save(p)
    assert ing.describe_visual(str(p)) == "bar chart: Q1=1"
    assert ing.describe_visual(str(p)) == "bar chart: Q1=1"
    assert len(calls) == 1


def test_vector_figure_detection(tmp_path):
    doc = fitz.open()
    pg = doc.new_page()
    pg.insert_text((72, 60), "Intro paragraph text here. " * 3)
    for i in range(6):
        pg.draw_rect(fitz.Rect(100 + i * 50, 400 - i * 30, 130 + i * 50, 500), fill=(0.2, 0.4, 0.8))
    pg.draw_line((95, 500), (420, 500))
    figs = ing._detect_figures(pg, [])
    assert len(figs) == 1
