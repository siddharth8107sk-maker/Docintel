"""Module 2a: build and persist the hybrid (dense + BM25) index.

CLI:  cd /home/claude && python -m docintel.retrieval.index [--chunks PATH] [--index-dir DIR]

Index layout (in index_dir):
  meta.json     backend info ("st" or "tfidf"), model name, chunk count, source mtime
  chunks.jsonl  copy of the chunks, row order == vector/bm25 order
  bm25.pkl      pickled rank_bm25.BM25Okapi
  vectors.npy   (backend == "st")   L2-normalised float32 embeddings
  tfidf.pkl     (backend == "tfidf") pickled {vectorizer, matrix}
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import re
import sys
from typing import List

from docintel.schemas import CHUNKS_PATH, INDEX_DIR, Chunk

log = logging.getLogger("docintel.retrieval")

DEFAULT_MODEL = "all-MiniLM-L6-v2"
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


def tokenize(text: str) -> List[str]:
    """Lowercase word/number tokenizer shared by indexing and querying."""
    return _TOKEN_RE.findall((text or "").lower())


def chunk_text(c: Chunk) -> str:
    """Text that gets embedded/indexed: section heading gives extra context."""
    return f"{c.section}\n{c.content}" if c.section else c.content


def load_chunks(path: str = CHUNKS_PATH) -> List[Chunk]:
    chunks: List[Chunk] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            chunks.append(Chunk(
                chunk_id=d["chunk_id"], doc_id=d["doc_id"], page=int(d["page"]),
                section=d.get("section", "") or "", type=d["type"],
                content=d.get("content", "") or "", page_image=d.get("page_image", "") or "",
                bbox=d.get("bbox"),
            ))
    return chunks


def load_embedder(model_name: str | None = None):
    """Return (SentenceTransformer, name) or (None, name) if unavailable (offline)."""
    name = model_name or os.environ.get("DOCINTEL_EMBED_MODEL", DEFAULT_MODEL)
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(name), name
    except Exception as e:  # network failure, missing package, bad model name...
        log.warning("Could not load embedding model %r (%s: %s). Falling back to a "
                    "TF-IDF substitute for dense retrieval (offline mode).",
                    name, type(e).__name__, str(e)[:150])
        return None, name


def build_index(chunks_path: str = CHUNKS_PATH, index_dir: str = INDEX_DIR) -> dict:
    import numpy as np
    chunks = load_chunks(chunks_path)
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}")
    os.makedirs(index_dir, exist_ok=True)
    texts = [chunk_text(c) for c in chunks]

    # BM25
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([tokenize(t) or ["_"] for t in texts])
    with open(os.path.join(index_dir, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25, f)

    # Dense (or TF-IDF fallback)
    for fn in ("vectors.npy", "tfidf.pkl"):  # drop stale artefacts from another backend
        p = os.path.join(index_dir, fn)
        if os.path.exists(p):
            os.remove(p)
    model, name = load_embedder()
    if model is not None:
        vecs = model.encode(texts, batch_size=32, normalize_embeddings=True,
                            show_progress_bar=False)
        np.save(os.path.join(index_dir, "vectors.npy"), np.asarray(vecs, dtype="float32"))
        backend = "st"
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import docintel.retrieval.index as _idx_mod
        vec = TfidfVectorizer(tokenizer=_idx_mod.tokenize, token_pattern=None, lowercase=False,
                              ngram_range=(1, 2), sublinear_tf=True)
        mat = vec.fit_transform(texts)
        with open(os.path.join(index_dir, "tfidf.pkl"), "wb") as f:
            pickle.dump({"vectorizer": vec, "matrix": mat}, f)
        backend = "tfidf"

    with open(os.path.join(index_dir, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    meta = {"backend": backend, "model": name if backend == "st" else "tfidf",
            "n_chunks": len(chunks), "source": os.path.abspath(chunks_path),
            "source_mtime": os.path.getmtime(chunks_path)}
    with open(os.path.join(index_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Indexed %d chunks into %s (dense backend: %s)", len(chunks), index_dir, meta["model"])
    return meta


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Build the docintel retrieval index")
    ap.add_argument("--chunks", default=CHUNKS_PATH)
    ap.add_argument("--index-dir", default=INDEX_DIR)
    a = ap.parse_args(argv)
    if not os.path.exists(a.chunks):
        print(f"chunks file not found: {a.chunks}", file=sys.stderr)
        return 1
    meta = build_index(a.chunks, a.index_dir)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
