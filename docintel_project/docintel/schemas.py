"""Shared contract between modules. Do not change field names without telling the other modules."""
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Literal

ChunkType = Literal["text", "table", "chart", "image", "ocr"]


@dataclass
class Chunk:
    """Produced by ingestion (Module 1), consumed by retrieval (Module 2)."""
    chunk_id: str            # e.g. "report2025_p12_t0"
    doc_id: str              # PDF file name, e.g. "report2025.pdf"
    page: int                # 1-based page number
    section: str             # nearest heading, "" if unknown
    type: ChunkType
    content: str             # text, table as markdown, or VLM caption/data summary
    page_image: str          # path to rendered page PNG, e.g. "data/pages/report2025_p12.png"
    bbox: Optional[List[float]] = None  # [x0, y0, x1, y1] in page coordinates

    def to_dict(self):
        return asdict(self)


@dataclass
class Citation:
    doc_id: str
    page: int
    section: str
    quote_or_value: str      # exact text/number from the source supporting the claim
    chunk_id: str = ""


@dataclass
class Answer:
    """Produced by reasoning (Module 3)."""
    answer: str
    citations: List[Citation] = field(default_factory=list)
    calculations: List[str] = field(default_factory=list)   # e.g. "(92-78)/78 = 17.95%"
    supported: bool = True   # False when evidence was insufficient (answer is a refusal)

    def to_dict(self):
        return asdict(self)


# Paths agreed by all modules
DATA_DIR = "data"
DOCS_DIR = "data/docs"            # input PDFs
PAGES_DIR = "data/pages"          # rendered page images
CHUNKS_PATH = "data/chunks.jsonl" # Module 1 output
INDEX_DIR = "data/index"          # Module 2 output
