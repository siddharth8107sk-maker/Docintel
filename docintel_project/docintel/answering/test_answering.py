import json
import os
from types import SimpleNamespace as NS

import pytest

from docintel.schemas import Chunk
from docintel.answering import answer as A
from docintel.answering.calculator import CalcError, evaluate, format_calculation

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


def load_chunks():
    out = []
    for d in json.load(open(os.path.join(FIX, "fake_chunks.json"))):
        d["page_image"] = os.path.join(FIX, os.path.basename(d["page_image"]))
        out.append(Chunk(**d))
    return out


class FakeClient:
    """Mock anthropic client: replays scripted responses and records requests."""
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []
        self.messages = NS(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        return self.responses.pop(0)


def tool(name, inp, id="t1"):
    return NS(type="tool_use", id=id, name=name, input=inp)


def resp(*blocks):
    return NS(content=list(blocks), stop_reason="tool_use")


def retriever_of(chunks):
    return lambda q: chunks


# ---------------- calculator
def test_calc_basic():
    assert evaluate("(92-78)/78") == pytest.approx(0.17948, abs=1e-4)
    assert evaluate("2**3 + 4*(5-1)") == 24
    assert evaluate("-3 + +5") == 2
    assert evaluate("round(10/3, 2)") == 3.33


def test_calc_pct_change():
    assert evaluate("pct_change(78, 92)") == pytest.approx(17.9487, abs=1e-3)
    assert evaluate("pct_change(100, 50)") == -50
    with pytest.raises(CalcError):
        evaluate("pct_change(0, 5)")
    assert format_calculation("pct_change(78,92)", evaluate("pct_change(78,92)")) == "pct_change(78,92) = 17.95%"
    assert format_calculation("40+52", 92) == "40+52 = 92"


@pytest.mark.parametrize("bad", ["__import__('os').system('ls')", "open('x')", "a+1", "().__class__",
                                 "[1,2]", "'a'*3", "2**9999", "1/0", "", "lambda: 1", "x=1", "True+1",
                                 "pct_change(1,2,3)", "abs.__call__(1)", "round(1, n=2)"])
def test_calc_rejects_unsafe(bad):
    with pytest.raises(CalcError):
        evaluate(bad)


# ---------------- answering
def good_submit(cites, supported=True, text="Revenue grew."):
    return tool("submit_answer", {"answer": text, "supported": supported, "citations": cites}, "s1")


def test_calculation_loop_and_valid_citation():
    cites = [{"doc_id": "fake.pdf", "page": 2, "section": "Financials", "quote_or_value": "2024 = 92 M USD"}]
    client = FakeClient([resp(tool("calculator", {"expression": "pct_change(78,92)"})),
                         resp(good_submit(cites, text="Revenue grew 17.95%."))])
    ans = A.answer_question("By what percent did revenue grow?", retriever_of(load_chunks()), client)
    assert ans.supported and ans.calculations == ["pct_change(78,92) = 17.95%"]
    assert ans.citations[0].chunk_id == "fake_p2_c0"
    # tool result was fed back to the model
    last = client.calls[1]["messages"][-1]["content"][0]
    assert last["type"] == "tool_result" and "17.95%" in last["content"]


def test_bad_calculator_input_is_reported_not_executed():
    client = FakeClient([resp(tool("calculator", {"expression": "__import__('os')"})),
                         resp(good_submit([{"doc_id": "fake.pdf", "page": 1, "quote_or_value": "1999"}]))])
    ans = A.answer_question("When was Acme founded?", retriever_of(load_chunks()), client)
    assert ans.calculations == []
    assert client.calls[1]["messages"][-1]["content"][0].get("is_error")


def test_bogus_citations_dropped():
    cites = [{"doc_id": "fake.pdf", "page": 99, "quote_or_value": "x"},
             {"doc_id": "other.pdf", "page": 1, "quote_or_value": "y"},
             {"doc_id": "fake.pdf", "page": 1, "section": "Overview", "quote_or_value": "founded in 1999"}]
    ans = A.answer_question("q", retriever_of(load_chunks()), FakeClient([resp(good_submit(cites))]))
    assert ans.supported and len(ans.citations) == 1 and ans.citations[0].page == 1
    assert "removed" in ans.answer


def test_no_valid_citation_means_unsupported():
    cites = [{"doc_id": "fake.pdf", "page": 99, "quote_or_value": "x"}]
    ans = A.answer_question("q", retriever_of(load_chunks()), FakeClient([resp(good_submit(cites))]))
    assert ans.supported is False and ans.citations == []
    assert "could not find supporting evidence" in ans.answer
    ans = A.answer_question("q", retriever_of(load_chunks()), FakeClient([resp(good_submit([]))]))
    assert ans.supported is False


def test_refusal_when_no_evidence_and_model_not_called():
    client = FakeClient([])
    ans = A.answer_question("anything", retriever_of([]), client)
    assert ans.supported is False and "could not find supporting evidence" in ans.answer
    assert client.calls == []


def test_model_declares_unsupported():
    sub = good_submit([], supported=False, text="The documents do not say.")
    ans = A.answer_question("q", retriever_of(load_chunks()), FakeClient([resp(sub)]))
    assert ans.supported is False and ans.answer == "The documents do not say."


def test_prose_reply_is_pushed_to_submit():
    prose = NS(content=[NS(type="text", text="It is 5.")], stop_reason="end_turn")
    sub = good_submit([{"doc_id": "fake.pdf", "page": 1, "quote_or_value": "1999"}])
    client = FakeClient([prose, resp(sub)])
    ans = A.answer_question("q", retriever_of(load_chunks()), client)
    assert ans.supported and len(client.calls) == 2


def test_page_images_included_and_deduped(monkeypatch):
    client = FakeClient([resp(good_submit([{"doc_id": "fake.pdf", "page": 2, "quote_or_value": "92"}]))])
    A.answer_question("q", retriever_of(load_chunks()), client)
    content = client.calls[0]["messages"][0]["content"]
    imgs = [b for b in content if b["type"] == "image"]
    assert len(imgs) == 2  # p2 chart + p3 (table & ocr share one image); text-only p1 has none
    assert all(b["source"]["type"] == "base64" and b["source"]["data"] for b in imgs)
    text = content[0]["text"]
    assert "doc_id=fake.pdf | page=2 | section=Financials" in text
    assert client.calls[0]["tools"][0]["name"] == "calculator"
    assert "never" in client.calls[0]["system"].lower() or "NEVER" in client.calls[0]["system"]


def test_image_cap(monkeypatch):
    base = load_chunks()[1]
    many = [Chunk(**{**base.to_dict(), "chunk_id": f"c{i}", "page": i}) for i in range(1, 10)]
    for i, c in enumerate(many):  # distinct image paths
        c.page_image = os.path.join(FIX, "fake_p2.png") if i == 0 else os.path.join(FIX, f"copy{i}.png")
    import shutil
    for i in range(1, 10):
        shutil.copy(os.path.join(FIX, "fake_p2.png"), os.path.join(FIX, f"copy{i}.png"))
    try:
        assert len(A.select_page_images(many, 4)) == 4
    finally:
        for i in range(1, 10):
            os.remove(os.path.join(FIX, f"copy{i}.png"))


def test_model_env(monkeypatch):
    monkeypatch.setenv("DOCINTEL_VLM_MODEL", "my-model")
    client = FakeClient([resp(good_submit([{"doc_id": "fake.pdf", "page": 1, "quote_or_value": "1999"}]))])
    A.answer_question("q", retriever_of(load_chunks()), client)
    assert client.calls[0]["model"] == "my-model"


def test_grouped_retrieval_used_for_comparison():
    ch = load_chunks()

    class R:
        def retrieve(self, q, k=8): return ch[:1]
        def retrieve_grouped(self, q, k_per_doc=4): return {"a.pdf": ch[:2], "b.pdf": ch[1:3]}
    assert len(A.retrieve_chunks("Compare revenue across both reports", R())) == 3
    assert len(A.retrieve_chunks("When was it founded?", R())) == 1


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(A.MissingAPIKeyError, match="ANTHROPIC_API_KEY"):
        A.answer_question("q", retriever_of(load_chunks()))
    # a question that retrieves real chunks must hit the key check -> clean exit code, no traceback
    assert A.main(["Compare production efficiency between Q2 and Q4"]) == 1
