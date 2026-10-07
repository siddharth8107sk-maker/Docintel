"""Module 2b: hybrid retrieval (BM25 + dense, fused with reciprocal rank fusion)."""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import pickle
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from docintel.schemas import CHUNKS_PATH, INDEX_DIR, PAGES_DIR, Chunk
from docintel.retrieval.index import build_index, load_chunks, load_embedder, tokenize

log = logging.getLogger("docintel.retrieval")

RRF_K = 60
# Chunk types that must carry a page image so the answerer can look at the page.
VISUAL_TYPES = ("table", "chart", "ocr", "image")
TABLE_CHART_BOOST = 1.5

# Words suggesting the answer lives in a table or chart.
_NUMERIC_RE = re.compile(
    r"\b(compare[ds]?|comparison|versus|vs\.?|trend|trends|chart|charts|graph|graphs|plot|figure|"
    r"figures|table|percent|percentage|%|increase[ds]?|decrease[ds]?|growth|change[ds]?|rose|fell|"
    r"decline[ds]?|highest|lowest|average|mean|total|sum|difference|ratio|rate|how (?:much|many)|"
    r"q[1-4]|quarter(?:ly)?|annual|yearly|monthly)\b|\d", re.I)
_CHART_RE = re.compile(r"\b(chart|charts|graph|graphs|plot|trend|trends|figure|visuali[sz]ation|curve|bar|line)\b", re.I)


def is_numeric_question(q: str) -> bool:
    return bool(_NUMERIC_RE.search(q or ""))


def _rank_scores(scores: np.ndarray, idx: np.ndarray) -> Dict[int, int]:
    """Map candidate index -> 1-based rank by descending score (stable)."""
    order = idx[np.argsort(-scores[idx], kind="stable")]
    return {int(i): r + 1 for r, i in enumerate(order)}


class Retriever:
    def __init__(self, index_dir: str = INDEX_DIR):
        self.index_dir = index_dir
        with open(os.path.join(index_dir, "meta.json")) as f:
            self.meta = json.load(f)
        self.chunks: List[Chunk] = load_chunks(os.path.join(index_dir, "chunks.jsonl"))
        with open(os.path.join(index_dir, "bm25.pkl"), "rb") as f:
            self.bm25 = pickle.load(f)
        self.backend = self.meta["backend"]
        self._model = None
        if self.backend == "st":
            self.vectors = np.load(os.path.join(index_dir, "vectors.npy"))
            self._model, _ = load_embedder(self.meta["model"])
            if self._model is None:  # index was built online but we are offline now
                raise RuntimeError(
                    f"Index was built with embedding model {self.meta['model']!r} which cannot be "
                    "loaded now; rebuild the index (python -m docintel.retrieval.index).")
        else:
            import __main__
            if not hasattr(__main__, "tokenize"):
                setattr(__main__, "tokenize", tokenize)
            with open(os.path.join(index_dir, "tfidf.pkl"), "rb") as f:
                d = pickle.load(f)
            self._vectorizer, self._matrix = d["vectorizer"], d["matrix"]
        self.doc_ids = sorted({c.doc_id for c in self.chunks})
        self._by_page: Dict[Tuple[str, int], List[int]] = {}
        for i, c in enumerate(self.chunks):
            self._by_page.setdefault((c.doc_id, c.page), []).append(i)

    # ---- scoring ---------------------------------------------------------
    def _dense_scores(self, question: str) -> np.ndarray:
        if self.backend == "st":
            q = self._model.encode([question], normalize_embeddings=True, show_progress_bar=False)[0]
            return self.vectors @ q
        qv = self._vectorizer.transform([question])
        return np.asarray((self._matrix @ qv.T).todense()).ravel()

    def _mask(self, doc_ids, types, pages) -> np.ndarray:
        def norm(x):
            if x is None:
                return None
            return {x} if isinstance(x, (str, int)) else set(x)
        d, t, p = norm(doc_ids), norm(types), norm(pages)
        return np.array([(d is None or c.doc_id in d) and (t is None or c.type in t)
                         and (p is None or c.page in p) for c in self.chunks], dtype=bool)

    def _with_image(self, c: Chunk) -> Chunk:
        """Copy of chunk, guaranteeing page_image for visual types."""
        if c.type in VISUAL_TYPES and not c.page_image:
            stem = os.path.splitext(os.path.basename(c.doc_id))[0]
            return dataclasses.replace(c, page_image=f"{PAGES_DIR}/{stem}_p{c.page}.png")
        return dataclasses.replace(c)

    def _search(self, question, k, doc_ids=None, types=None, pages=None):
        """Return list of (idx, fused_score, bm25_score, dense_score)."""
        mask = self._mask(doc_ids, types, pages)
        idx = np.flatnonzero(mask)
        if idx.size == 0 or not question.strip():
            return []
        bm = np.asarray(self.bm25.get_scores(tokenize(question) or ["_"]), dtype=float)
        dn = self._dense_scores(question)
        # Only rank BM25 on chunks that actually matched; ties at zero carry no signal.
        bm_idx = idx[bm[idx] > 0]
        r_bm = _rank_scores(bm, bm_idx)
        r_dn = _rank_scores(dn, idx[dn[idx] > 0]) if self.backend == "tfidf" else _rank_scores(dn, idx)
        fused = {int(i): 0.0 for i in idx}
        for r in (r_bm, r_dn):
            for i, rank in r.items():
                fused[i] += 1.0 / (RRF_K + rank)
        if is_numeric_question(question):
            chart_q = bool(_CHART_RE.search(question))
            for i in fused:
                t = self.chunks[i].type
                if t == "table" or t == "chart":
                    fused[i] *= TABLE_CHART_BOOST
                if chart_q and t == "chart":
                    fused[i] *= 1.2
        ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [(i, s, float(bm[i]), float(dn[i])) for i, s in ranked if s > 0]

    # ---- public API ------------------------------------------------------
    def retrieve_with_scores(self, question: str, k: int = 8, doc_ids=None, types=None,
                             pages=None) -> List[Tuple[Chunk, float]]:
        return [(self._with_image(self.chunks[i]), s)
                for i, s, _, _ in self._search(question, k, doc_ids, types, pages)]

    def retrieve(self, question: str, k: int = 8, doc_ids=None, types=None, pages=None) -> List[Chunk]:
        """Top-k chunks. doc_ids/types/pages accept a single value or an iterable."""
        return [c for c, _ in self.retrieve_with_scores(question, k, doc_ids, types, pages)]

    def retrieve_grouped(self, question: str, k_per_doc: int = 4, doc_ids=None, types=None,
                         pages=None) -> Dict[str, List[Chunk]]:
        """Retrieve per document and merge, so every relevant document is represented.

        A document counts as relevant if its best chunk has a lexical (BM25) match or, failing
        that, if no document has one (then all documents are returned). Dict order = documents
        sorted by best fused score.
        """
        per_doc = {}
        for d in self.doc_ids:
            if doc_ids is not None and d not in ({doc_ids} if isinstance(doc_ids, str) else set(doc_ids)):
                continue
            hits = self._search(question, k_per_doc, d, types, pages)
            if hits:
                per_doc[d] = hits
        lexical = {d: h for d, h in per_doc.items() if any(x[2] > 0 for x in h)}
        chosen = lexical or per_doc
        out = {}
        for d, h in sorted(chosen.items(), key=lambda kv: -kv[1][0][1]):
            out[d] = [self._with_image(self.chunks[i]) for i, *_ in h]
        return out

    def neighbors(self, chunk: Chunk, window: int = 1, include_self: bool = True) -> List[Chunk]:
        """Chunks of the same document on the same page (window=1), in index order.

        window=n additionally includes pages within +/-(n-1) of chunk.page, so window=2 pulls in
        the previous and next page too. Set include_self=False to omit `chunk` itself.
        """
        reach = max(window - 1, 0)
        out = []
        for p in range(chunk.page - reach, chunk.page + reach + 1):
            for i in self._by_page.get((chunk.doc_id, p), []):
                c = self.chunks[i]
                if include_self or c.chunk_id != chunk.chunk_id:
                    out.append(self._with_image(c))
        return out


# ---- module-level convenience ----------------------------------------------
_default: Optional[Retriever] = None
_default_dir: Optional[str] = None


def get_retriever(index_dir: str = INDEX_DIR, chunks_path: str = CHUNKS_PATH,
                  rebuild: bool = False) -> Retriever:
    """Lazy singleton. Builds the index if missing or older than chunks_path (when it exists)."""
    global _default, _default_dir
    meta_path = os.path.join(index_dir, "meta.json")
    have_chunks = os.path.exists(chunks_path)
    stale = False
    if os.path.exists(meta_path) and have_chunks:
        stale = os.path.getmtime(chunks_path) > json.load(open(meta_path)).get("source_mtime", 0) + 1e-6
    if rebuild or stale or not os.path.exists(meta_path):
        if not have_chunks:
            raise FileNotFoundError(f"No index at {index_dir} and no chunks at {chunks_path}")
        build_index(chunks_path, index_dir)
        _default = None
    if _default is None or _default_dir != index_dir:
        _default, _default_dir = Retriever(index_dir), index_dir
    return _default


def retrieve(question: str, k: int = 8, **filters) -> List[Chunk]:
    return get_retriever().retrieve(question, k, **filters)


def retrieve_with_scores(question: str, k: int = 8, **filters) -> List[Tuple[Chunk, float]]:
    return get_retriever().retrieve_with_scores(question, k, **filters)


def retrieve_grouped(question: str, k_per_doc: int = 4, **filters) -> Dict[str, List[Chunk]]:
    return get_retriever().retrieve_grouped(question, k_per_doc, **filters)


def neighbors(chunk: Chunk, window: int = 1, include_self: bool = True) -> List[Chunk]:
    return get_retriever().neighbors(chunk, window, include_self)
