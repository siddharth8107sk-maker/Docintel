"""Module 3: grounded answering with citations, page images and a Python calculator tool.

Public API:
    answer_question(question, retriever=None, client=None) -> Answer
    answer_with_context(question, retriever=None, client=None) -> (Answer, list[Chunk])
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
from typing import Callable, List, Optional, Tuple

from docintel.schemas import Answer, Chunk, Citation
from docintel.answering.calculator import CalcError, evaluate, format_calculation

DEFAULT_MODEL = "claude-sonnet-4-5"
VISUAL_TYPES = {"chart", "table", "image", "ocr"}
REFUSAL_TEXT = "I could not find supporting evidence in the documents to answer this question."
MAX_TOOL_ROUNDS = 8

SYSTEM_PROMPT = """You are a careful document question-answering assistant.

Rules (strict):
1. Answer ONLY from the evidence provided (extracted text chunks and page images). Never use outside knowledge.
2. Every claim must be backed by a citation: doc_id, page, section, and an exact quote_or_value copied from the source.
3. Report numbers exactly as you read them (keep units, signs and decimals). When a chart, table or scanned page is provided as an image, read the values from the image and cross-check them with the extracted text; if they disagree, say so.
4. List every number used in a calculation, with its citation.
5. NEVER do arithmetic in prose or in your head. Use the `calculator` tool for every sum, difference, ratio, percentage or percent change (use pct_change(old,new)) and quote the tool's result.
6. If the evidence is insufficient or does not contain the answer, say so plainly and call submit_answer with supported=false and no invented citations.
7. When finished, call `submit_answer` exactly once with the final answer, citations and supported flag. Only cite (doc_id, page) pairs that appear in the provided evidence."""

CALCULATOR_TOOL = {
    "name": "calculator",
    "description": ("Evaluate an arithmetic expression exactly in Python. Supports + - * / ** parentheses, "
                    "pct_change(old,new) (returns percent), round(x,n), abs, min, max, sqrt. "
                    "Use for ALL arithmetic."),
    "input_schema": {"type": "object",
                     "properties": {"expression": {"type": "string",
                                                   "description": "e.g. (92-78)/78*100 or pct_change(78,92)"}},
                     "required": ["expression"]},
}

SUBMIT_TOOL = {
    "name": "submit_answer",
    "description": "Submit the final grounded answer with citations. Call exactly once when done.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "supported": {"type": "boolean",
                          "description": "false if the documents do not contain enough evidence"},
            "citations": {"type": "array", "items": {
                "type": "object",
                "properties": {"doc_id": {"type": "string"}, "page": {"type": "integer"},
                               "section": {"type": "string"},
                               "quote_or_value": {"type": "string",
                                                  "description": "exact text or number from the source"}},
                "required": ["doc_id", "page", "quote_or_value"]}},
        },
        "required": ["answer", "supported", "citations"],
    },
}


class MissingAPIKeyError(RuntimeError):
    pass


# ----------------------------------------------------------------- retrieval
_MULTI_PAT = re.compile(
    r"\b(compare|compared|comparison|versus|vs\.?|both|across|between|each (document|report|source)|"
    r"all (documents|reports|sources)|documents|reports|sources|differ\w*|contrast)\b", re.I)


def wants_grouped(question: str) -> bool:
    return bool(_MULTI_PAT.search(question))


def _default_retriever() -> Callable:
    try:
        from docintel.retrieval import retriever as r  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Retrieval module (docintel.retrieval.retriever) is not available yet "
                           f"({e}). Pass a retriever=... callable or build the index first.")
    return r  # module with retrieve / retrieve_grouped


def retrieve_chunks(question: str, retriever=None) -> List[Chunk]:
    """retriever may be a module/object exposing retrieve[/retrieve_grouped], or a plain callable(question)->list."""
    if retriever is None:
        retriever = _default_retriever()
    grouped = wants_grouped(question)
    if grouped and hasattr(retriever, "retrieve_grouped"):
        groups = retriever.retrieve_grouped(question, k_per_doc=4)
        out, seen = [], set()
        for chunks in groups.values():
            for c in chunks:
                if c.chunk_id not in seen:
                    seen.add(c.chunk_id)
                    out.append(c)
        return out
    if hasattr(retriever, "retrieve"):
        return list(retriever.retrieve(question, k=8))
    return list(retriever(question))


# ------------------------------------------------------------ message build
_MEDIA = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
          ".gif": "image/gif", ".webp": "image/webp"}


def image_block(path: str) -> Optional[dict]:
    ext = os.path.splitext(path)[1].lower()
    if ext not in _MEDIA or not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": _MEDIA[ext], "data": data}}


def select_page_images(chunks: List[Chunk], max_images: Optional[int] = None) -> List[Tuple[Chunk, dict]]:
    """Deduplicated page-image blocks for chart/table/image/ocr chunks, in retrieval order, capped."""
    if max_images is None:
        max_images = int(os.environ.get("DOCINTEL_MAX_IMAGES", "5"))
    out, seen = [], set()
    for c in chunks:
        if c.type not in VISUAL_TYPES or not c.page_image or c.page_image in seen:
            continue
        blk = image_block(c.page_image)
        if blk is None:
            continue
        seen.add(c.page_image)
        out.append((c, blk))
        if len(out) >= max_images:
            break
    return out


def chunk_label(c: Chunk) -> str:
    return f"[{c.chunk_id}] doc_id={c.doc_id} | page={c.page} | section={c.section or 'unknown'} | type={c.type}"


def build_user_content(question: str, chunks: List[Chunk]) -> list:
    parts = ["EVIDENCE (extracted text; image-type chunk text is a machine caption and may be imperfect):"]
    for c in chunks:
        parts.append(f"{chunk_label(c)}\n{c.content}")
    content: list = [{"type": "text", "text": "\n\n".join(parts)}]
    for c, blk in select_page_images(chunks):
        content.append({"type": "text", "text": f"Page image for doc_id={c.doc_id} page={c.page} ({c.type} chunk):"})
        content.append(blk)
    content.append({"type": "text", "text": f"QUESTION: {question}\n\nUse the calculator tool for any arithmetic, "
                                            "then call submit_answer."})
    return content


# ------------------------------------------------------- citation validation
def _norm_doc(d: str) -> str:
    return os.path.basename(str(d)).strip().lower()


def validate_citations(raw: list, chunks: List[Chunk]) -> Tuple[List[Citation], List[dict]]:
    """Keep citations whose (doc_id, page) exist among retrieved chunks and have a quote. Returns (valid, dropped)."""
    index = {}
    for c in chunks:
        index.setdefault((_norm_doc(c.doc_id), c.page), []).append(c)
    valid, dropped, seen = [], [], set()
    for r in raw or []:
        try:
            doc, page = str(r.get("doc_id", "")), int(r.get("page"))
        except (TypeError, ValueError, AttributeError):
            dropped.append(r)
            continue
        quote = str(r.get("quote_or_value", "")).strip()
        matches = index.get((_norm_doc(doc), page))
        if not matches or not quote:
            dropped.append(r)
            continue
        section = str(r.get("section") or "") or matches[0].section
        key = (_norm_doc(doc), page, quote)
        if key in seen:
            continue
        seen.add(key)
        # prefer the chunk that actually contains the quote
        best = next((m for m in matches if quote.lower() in m.content.lower()), matches[0])
        valid.append(Citation(doc_id=matches[0].doc_id, page=page, section=section or best.section,
                              quote_or_value=quote, chunk_id=best.chunk_id))
    return valid, dropped


def refusal(text: str = REFUSAL_TEXT) -> Answer:
    return Answer(answer=text, citations=[], calculations=[], supported=False)


# ------------------------------------------------------------------- client
def make_client():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise MissingAPIKeyError("ANTHROPIC_API_KEY is not set. Export it first, e.g. "
                                 "`export ANTHROPIC_API_KEY=sk-ant-...`, then retry.")
    import anthropic
    return anthropic.Anthropic()


def _get(block, name, default=None):
    return block.get(name, default) if isinstance(block, dict) else getattr(block, name, default)


def _serialize_assistant(content) -> list:
    out = []
    for b in content:
        t = _get(b, "type")
        if t == "text":
            out.append({"type": "text", "text": _get(b, "text", "") or "(no text)"})
        elif t == "tool_use":
            out.append({"type": "tool_use", "id": _get(b, "id"), "name": _get(b, "name"),
                        "input": _get(b, "input", {})})
    return out


# --------------------------------------------------------------------- main
def answer_with_context(question: str, retriever=None, client=None) -> Tuple[Answer, List[Chunk]]:
    chunks = retrieve_chunks(question, retriever)
    if not chunks:
        return refusal(), []

    use_local = os.environ.get("USE_LOCAL_ML", "1") == "1"
    if use_local and client is None:
        from docintel.answering.local_ml import local_ml_answer
        return local_ml_answer(question, chunks), chunks

    try:
        if client is None:
            client = make_client()
    except Exception:
        from docintel.answering.local_ml import local_ml_answer
        return local_ml_answer(question, chunks), chunks

    model = os.environ.get("DOCINTEL_VLM_MODEL", DEFAULT_MODEL)

    messages = [{"role": "user", "content": build_user_content(question, chunks)}]
    calculations: List[str] = []
    submitted = None

    try:
        for round_no in range(MAX_TOOL_ROUNDS + 1):
            force = round_no >= MAX_TOOL_ROUNDS
            kwargs = dict(model=model, max_tokens=2048, system=SYSTEM_PROMPT,
                          tools=[CALCULATOR_TOOL, SUBMIT_TOOL], messages=messages)
            if force:
                kwargs["tool_choice"] = {"type": "tool", "name": "submit_answer"}
            resp = client.messages.create(**kwargs)
            blocks = list(resp.content)
            tool_uses = [b for b in blocks if _get(b, "type") == "tool_use"]
            sub = next((b for b in tool_uses if _get(b, "name") == "submit_answer"), None)

            results = []
            for tu in tool_uses:
                if _get(tu, "name") != "calculator":
                    continue
                expr = (_get(tu, "input", {}) or {}).get("expression", "")
                try:
                    val = evaluate(expr)
                    line = format_calculation(expr, val)
                    calculations.append(line)
                    results.append({"type": "tool_result", "tool_use_id": _get(tu, "id"), "content": line})
                except CalcError as e:
                    results.append({"type": "tool_result", "tool_use_id": _get(tu, "id"),
                                    "content": f"Error: {e}", "is_error": True})
            if sub is not None:
                submitted = _get(sub, "input", {}) or {}
                break
            if not tool_uses:
                # model answered in prose; demand structured output
                messages.append({"role": "assistant", "content": _serialize_assistant(blocks)})
                messages.append({"role": "user", "content": "Now call submit_answer with the final structured answer."})
                continue
            messages.append({"role": "assistant", "content": _serialize_assistant(blocks)})
            messages.append({"role": "user", "content": results})
    except Exception as e:
        from docintel.answering.local_ml import local_ml_answer
        return local_ml_answer(question, chunks), chunks

    if submitted is None:
        return refusal("The model did not return a structured answer."), chunks

    text = str(submitted.get("answer", "")).strip()
    claimed_supported = bool(submitted.get("supported", False))
    valid, dropped = validate_citations(submitted.get("citations", []), chunks)

    if not claimed_supported:
        return Answer(answer=text or REFUSAL_TEXT, citations=valid, calculations=calculations,
                      supported=False), chunks
    if not valid:
        return refusal(), chunks
    if dropped:
        text += f"\n\n(Note: {len(dropped)} citation(s) were removed because they did not match retrieved pages.)"
    return Answer(answer=text, citations=valid, calculations=calculations, supported=True), chunks


def answer_question(question: str, retriever=None, client=None) -> Answer:
    return answer_with_context(question, retriever, client)[0]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print('usage: python -m docintel.answering.answer "your question"')
        return 2
    try:
        ans = answer_question(" ".join(argv))
    except MissingAPIKeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # clean message instead of a stack trace
        print(f"Error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(json.dumps(ans.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
