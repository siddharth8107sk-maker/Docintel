import os
import pytest

from docintel.retrieval.index import build_index
from docintel.retrieval.retriever import Retriever
from docintel.schemas import Chunk

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "fake_chunks.jsonl")


@pytest.fixture(scope="module")
def r(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("idx"))
    build_index(FIX, d)
    return Retriever(d)


def test_returns_chunks(r):
    res = r.retrieve("production efficiency", k=5)
    assert 0 < len(res) <= 5 and all(isinstance(c, Chunk) for c in res)


def test_filters(r):
    q = "efficiency and downtime in Q3"
    assert {c.doc_id for c in r.retrieve(q, k=10, doc_ids=["maintenance_review_2024.pdf"])} == {"maintenance_review_2024.pdf"}
    assert {c.type for c in r.retrieve(q, k=10, types=["table"])} == {"table"}
    assert {c.page for c in r.retrieve(q, k=10, pages=[2])} == {2}
    res = r.retrieve(q, k=10, doc_ids="plant_report_2024.pdf", types=["text", "chart"], pages=[3])
    assert res and all(c.doc_id == "plant_report_2024.pdf" and c.page == 3 and c.type in ("text", "chart") for c in res)


def test_grouped_multiple_docs(r):
    g = r.retrieve_grouped("how did efficiency and downtime change from Q1 to Q4?", k_per_doc=3)
    assert set(g) == {"plant_report_2024.pdf", "maintenance_review_2024.pdf"}
    assert all(1 <= len(v) <= 3 and all(c.doc_id == d for c in v) for d, v in g.items())


def test_numeric_question_surfaces_table(r):
    res = r.retrieve("Compare the production efficiency percent in Q1 and Q4", k=3)
    assert any(c.type == "table" and c.doc_id == "plant_report_2024.pdf" for c in res)
    res = r.retrieve("What were the downtime hours in Q3 and Q4?", k=3)
    assert any(c.type == "table" and c.doc_id == "maintenance_review_2024.pdf" for c in res)


def test_chart_question_surfaces_chart(r):
    res = r.retrieve("What does the chart show about the downtime trend?", k=3)
    assert any(c.type == "chart" and c.doc_id == "maintenance_review_2024.pdf" for c in res)
    res = r.retrieve("graph of quarterly efficiency trend", k=3)
    assert any(c.type == "chart" and c.doc_id == "plant_report_2024.pdf" for c in res)


def test_page_image_populated_for_visual_types(r):
    # fixture deliberately leaves page_image empty on one table and one ocr chunk
    res = r.retrieve("downtime hours per quarter table; bearing replaced press 3", k=12)
    visual = [c for c in res if c.type in ("table", "chart", "ocr")]
    assert visual and all(c.page_image.endswith(".png") for c in visual)
    assert any(c.chunk_id == "maintenance_review_2024_p2_t0" and c.page_image == "data/pages/maintenance_review_2024_p2.png" for c in visual)


def test_neighbors_same_page(r):
    table = next(c for c in r.chunks if c.chunk_id == "plant_report_2024_p2_t0")
    ids = {c.chunk_id for c in r.neighbors(table)}
    assert ids == {"plant_report_2024_p2_x0", "plant_report_2024_p2_t0"}
    assert "plant_report_2024_p2_t0" not in {c.chunk_id for c in r.neighbors(table, include_self=False)}
    assert {c.page for c in r.neighbors(table, window=2)} == {1, 2, 3}


def test_scores_sorted(r):
    res = r.retrieve_with_scores("efficiency", k=5)
    s = [x for _, x in res]
    assert s == sorted(s, reverse=True)
