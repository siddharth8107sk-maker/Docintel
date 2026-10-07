"""Evaluate the QA pipeline on eval_questions.json -> data/eval_report.md.

Usage: cd /home/claude && python -m docintel.answering.evaluate [--limit N] [--out data/eval_report.md]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

from docintel.answering.answer import MissingAPIKeyError, answer_with_context

QUESTIONS = os.path.join(os.path.dirname(__file__), "eval_questions.json")
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def numbers_in(text: str):
    out = []
    for m in _NUM.findall(text or ""):
        try:
            out.append(float(m.replace(",", "").rstrip(".")))
        except ValueError:
            pass
    return out


def value_matches(expected, answer_text: str) -> bool:
    """Numbers match within 0.5% (or 0.01 abs); strings match case-insensitively."""
    if isinstance(expected, (int, float)) or re.fullmatch(r"-?[\d,]*\.?\d+%?", str(expected).strip()):
        e = float(str(expected).replace(",", "").rstrip("%"))
        return any(abs(n - e) <= max(0.01, abs(e) * 0.005) for n in numbers_in(answer_text))
    return str(expected).lower() in (answer_text or "").lower()


def is_placeholder(q) -> bool:
    return any(str(v).startswith("PLACEHOLDER") for v in q.get("expected_values", []))


def score(q, ans):
    r = {"id": q["id"], "category": q["category"], "supported": ans.supported, "n_cites": len(ans.citations)}
    unans = q.get("unanswerable", False)
    r["refusal_correct"] = (not ans.supported) if unans else None
    if not unans:
        r["citation_present"] = ans.supported and len(ans.citations) > 0
        if q.get("expected_page") is not None:
            r["page_correct"] = any(c.page == q["expected_page"] and
                                    (not q.get("expected_doc") or c.doc_id == q["expected_doc"])
                                    for c in ans.citations)
        vals = q.get("expected_values") or []
        if vals and not is_placeholder(q):
            hay = ans.answer + " " + " ".join(ans.calculations)
            r["numeric_match"] = all(value_matches(v, hay) for v in vals)
    return r


def rate(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return (sum(vals), len(vals)) if vals else (0, 0)


def fmt_rate(rows, key):
    n, d = rate(rows, key)
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "n/a"


def run(limit=None, out_path="data/eval_report.md", retriever=None, client=None):
    spec = json.load(open(QUESTIONS))
    qs = spec["questions"][:limit] if limit else spec["questions"]
    rows, details = [], []
    for q in qs:
        t0 = time.time()
        try:
            ans, _ = answer_with_context(q["question"], retriever, client)
        except (MissingAPIKeyError, RuntimeError):
            raise  # infrastructure problem (no key / no retriever): abort, don't score as refusal
        except Exception as e:
            print(f"[{q['id']}] error: {e}", file=sys.stderr)
            from docintel.schemas import Answer
            ans = Answer(answer=f"ERROR: {e}", supported=False)
        r = score(q, ans)
        r["secs"] = round(time.time() - t0, 1)
        rows.append(r)
        details.append((q, ans))
        print(f"[{q['id']}] {q['category']}: supported={ans.supported} cites={len(ans.citations)}")

    L = ["# Evaluation report", "", f"Questions: {len(rows)}", ""]
    if spec.get("_note"):
        L += [f"> Note: {spec['_note']}", ""]
    L += ["| Metric | Result |", "|---|---|",
          f"| Citation present rate (answerable) | {fmt_rate(rows, 'citation_present')} |",
          f"| Page-correct rate (expected_page given) | {fmt_rate(rows, 'page_correct')} |",
          f"| Numeric match (real expected values) | {fmt_rate(rows, 'numeric_match')} |",
          f"| Refusal-correct rate (unanswerable) | {fmt_rate(rows, 'refusal_correct')} |", "",
          "## Per question", "",
          "| id | category | supported | cites | citation | page | numeric | refusal ok | secs |",
          "|---|---|---|---|---|---|---|---|---|"]
    f = lambda v: "-" if v is None else ("yes" if v else "NO")
    for r in rows:
        L.append(f"| {r['id']} | {r['category']} | {r['supported']} | {r['n_cites']} | {f(r.get('citation_present'))} | "
                 f"{f(r.get('page_correct'))} | {f(r.get('numeric_match'))} | {f(r.get('refusal_correct'))} | {r['secs']} |")
    L += ["", "## Answers", ""]
    for q, a in details:
        L.append(f"**{q['id']}** ({q['category']}): {q['question']}  ")
        L.append(f"> {a.answer.replace(chr(10), ' ')}  ")
        if a.calculations:
            L.append(f"> calculations: {'; '.join(a.calculations)}  ")
        L.append("")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    open(out_path, "w").write("\n".join(L))
    print(f"Report written to {out_path}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default="data/eval_report.md")
    a = ap.parse_args()
    try:
        run(a.limit, a.out)
    except (MissingAPIKeyError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
