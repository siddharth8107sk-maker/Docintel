# Multimodal Document Intelligence

A Multimodal Document Intelligence and Question Answering System that can process documents containing text, tables, charts, images, and scanned pages, retrieve relevant evidence, and generate answers with document-level and page-level citations.

The project is designed around a Retrieval-Augmented Generation (RAG) architecture with support for OCR, visual document understanding, hybrid retrieval, calculations, and citation validation.

---

## Features

* Upload and process PDF, TXT, and Markdown documents
* Extract text from normal and scanned PDFs
* Render PDF pages as images
* Process visual content such as:

  * Tables
  * Charts
  * Graphs
  * Images
  * Scanned pages
* OCR support using Tesseract
* Intelligent document chunking
* Hybrid document retrieval using:

  * BM25
  * Sentence Transformers
  * TF-IDF fallback
* Question answering using retrieved document evidence
* Local ML answering fallback
* Optional Anthropic API integration
* Calculator support for numerical questions
* Citation validation
* Page-level source references
* Display of cited document pages
* Interactive Streamlit web interface
* Unit and module-level tests

---

## System Architecture

```text
                    +---------------------+
                    |     User Uploads    |
                    |   PDF / TXT / MD    |
                    +----------+----------+
                               |
                               v
                    +---------------------+
                    |    Ingestion Layer  |
                    |                     |
                    | - PDF extraction    |
                    | - OCR               |
                    | - Tables            |
                    | - Images            |
                    | - Charts            |
                    +----------+----------+
                               |
                               v
                    +---------------------+
                    |  Document Chunks    |
                    |   chunks.jsonl      |
                    +----------+----------+
                               |
                               v
              +---------------------------------+
              |       Retrieval Layer           |
              |                                 |
              |   +----------+  +----------+   |
              |   |   BM25   |  | TF-IDF / |   |
              |   | Retrieval|  | Embedding|   |
              |   +----+-----+  +----+-----+   |
              |        +----------+            |
              |                   |             |
              |                   v             |
              |           Hybrid Ranking       |
              +-----------------+---------------+
                                |
                                v
                    +---------------------+
                    |   Relevant Evidence |
                    |   + Page Images     |
                    +----------+----------+
                               |
                               v
                    +---------------------+
                    |   Answering Layer   |
                    |                     |
                    | - Local ML          |
                    | - Anthropic API     |
                    | - Calculator        |
                    | - Citation checking |
                    +----------+----------+
                               |
                               v
                    +---------------------+
                    |    Streamlit UI     |
                    |                     |
                    | Answer + Citations  |
                    | Calculations        |
                    | Cited Page Images   |
                    +---------------------+
```

---

## Project Structure

```text
docintel_project/
│
├── .env
│
├── docintel/
│   ├── __init__.py
│   ├── app.py
│   ├── schemas.py
│   ├── make_sample_docs.py
│   ├── requirements.txt
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── ingest.py
│   │   ├── uploader.py
│   │   └── test_ingestion.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── index.py
│   │   ├── retriever.py
│   │   ├── test_retrieval.py
│   │   └── fixtures/
│   │
│   └── answering/
│       ├── __init__.py
│       ├── answer.py
│       ├── calculator.py
│       ├── evaluate.py
│       ├── local_ml.py
│       ├── eval_questions.json
│       ├── test_answering.py
│       └── fixtures/
│
├── data/
│   ├── docs/
│   │   ├── PDF files
│   │   └── TXT files
│   │
│   ├── pages/
│   │   └── rendered page images
│   │
│   ├── index/
│   │   ├── bm25.pkl
│   │   ├── tfidf.pkl
│   │   ├── chunks.jsonl
│   │   └── meta.json
│   │
│   └── chunks.jsonl
│
└── scratch/
    └── development/testing files
```

---

# Processing Pipeline

## 1. Document Ingestion

Documents are placed in:

```text
data/docs/
```

The ingestion module processes the documents and extracts:

* Text
* Headings
* Tables
* Images
* Charts
* OCR text
* Page information
* Bounding boxes where available

The main ingestion logic is implemented in:

```text
docintel/ingestion/ingest.py
```

Uploaded documents are handled through:

```text
docintel/ingestion/uploader.py
```

---

## 2. OCR Processing

Scanned documents that do not contain machine-readable text can be processed using:

```text
pytesseract
```

The system also performs image preprocessing before OCR when necessary.

This allows the application to work with scanned examination papers, reports, receipts, and other image-based documents.

---

## 3. Visual Document Processing

PDF pages can be rendered into images.

The system keeps page images in:

```text
data/pages/
```

Visual chunks can represent:

```text
text
table
chart
image
ocr
```

This allows the answering system to use both extracted text and visual page information.

---

# Retrieval System

The project uses a hybrid retrieval approach.

## BM25

BM25 is used for keyword-based retrieval.

It is particularly useful when the question contains:

* Exact terms
* Names
* Numbers
* Technical keywords
* Document-specific terminology

The BM25 index is stored in:

```text
data/index/bm25.pkl
```

---

## Sentence Transformer Retrieval

The project can use a Sentence Transformer model:

```text
all-MiniLM-L6-v2
```

for semantic retrieval.

If the embedding model is unavailable, the system automatically falls back to TF-IDF.

---

## TF-IDF Fallback

The project supports TF-IDF as an offline-friendly retrieval alternative.

The TF-IDF index is stored in:

```text
data/index/tfidf.pkl
```

---

## Hybrid Ranking

Retrieved results are combined using ranking techniques so that relevant chunks can be selected from different retrieval signals.

The retrieval module is located at:

```text
docintel/retrieval/retriever.py
```

---

# Answering System

The answering layer receives:

```text
User Question
      +
Retrieved Chunks
      +
Relevant Page Images
```

and generates an answer based on the available evidence.

Main file:

```text
docintel/answering/answer.py
```

---

## Local ML Mode

The project supports a local answering mechanism through:

```text
docintel/answering/local_ml.py
```

The application can use local ML when configured, allowing the project to operate without making an external LLM request for every question.

---

## Anthropic API

The project also supports Anthropic for model-based answering.

The environment variable used is:

```text
ANTHROPIC_API_KEY
```

Create a `.env` file containing:

```env
ANTHROPIC_API_KEY=your_api_key_here
```

Never upload your real API key to GitHub.

Add `.env` to `.gitignore`.

---

# Calculator Support

Numerical questions can require calculations such as:

```text
percentage increase
percentage decrease
average
difference
ratio
```

The answering system includes a calculator module:

```text
docintel/answering/calculator.py
```

This allows arithmetic operations to be handled separately rather than relying entirely on language-model arithmetic.

---

# Citation System

One of the main features of this project is evidence-based answering.

Each answer can contain citations with:

```text
Document
Page
Section
Quote / Value
Chunk ID
```

Example:

```text
Document: Software Engineering Test.pdf
Page: 2
Section: Software Process Models
Quote: "The waterfall model..."
```

The system validates citations against the retrieved chunks before displaying them.

This reduces the possibility of generating citations that do not exist in the retrieved evidence.

---

# Streamlit Interface

The application provides a web interface using Streamlit.

The main application is:

```text
docintel/app.py
```

The interface allows users to:

1. Upload documents
2. Automatically index documents
3. View indexed documents
4. Ask questions
5. View generated answers
6. View citations
7. View calculation steps
8. View cited document pages

---

# Installation

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd docintel_project
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r docintel/requirements.txt
```

The project uses libraries including:

* PyMuPDF
* pdfplumber
* pytesseract
* Pillow
* ChromaDB
* rank-bm25
* sentence-transformers
* Anthropic
* Streamlit
* ReportLab
* Matplotlib
* Pandas
* Pytest

---

# Environment Configuration

Create:

```text
.env
```

Example:

```env
ANTHROPIC_API_KEY=your_api_key_here
```

Optional model configuration can also be supplied through environment variables used by the application:

```env
DOCINTEL_EMBED_MODEL=all-MiniLM-L6-v2
DOCINTEL_VLM_MODEL=your_model_name
USE_LOCAL_ML=1
DOCINTEL_MAX_IMAGES=5
```

---

# Running the Application

From the project root:

```bash
streamlit run docintel/app.py
```

The Streamlit interface will open in your browser.

You can then upload:

```text
.pdf
.txt
.md
```

documents and ask questions about them.

---

# Building the Retrieval Index

If the document chunks need to be re-indexed:

```bash
python -m docintel.retrieval.index
```

The generated index is stored in:

```text
data/index/
```

The index contains information such as:

```text
meta.json
chunks.jsonl
bm25.pkl
tfidf.pkl
```

Depending on whether a Sentence Transformer model is available, the project can instead generate:

```text
vectors.npy
```

---

# Testing

The project contains tests for the major modules.

### Ingestion Tests

```bash
pytest docintel/ingestion/test_ingestion.py
```

### Retrieval Tests

```bash
pytest docintel/retrieval/test_retrieval.py
```

### Answering Tests

```bash
pytest docintel/answering/test_answering.py
```

### Run All Tests

```bash
pytest
```

---

# Current Project Data

The project contains a prepared document collection under:

```text
data/docs/
```

and a pre-built retrieval index under:

```text
data/index/
```

The project index contains document chunks and retrieval metadata.

The documents also have rendered page images stored under:

```text
data/pages/
```

---

# Core Data Model

The system uses three important data structures.

## Chunk

A document chunk contains:

```text
chunk_id
doc_id
page
section
type
content
page_image
bbox
```

---

## Citation

A citation contains:

```text
doc_id
page
section
quote_or_value
chunk_id
```

---

## Answer

An answer contains:

```text
answer
citations
calculations
supported
```

This provides a consistent interface between ingestion, retrieval, and answering modules.

---

# Security

Do not commit secrets to GitHub.

Especially avoid committing:

```text
.env
.venv/
API keys
access tokens
passwords
private credentials
```

Recommended `.gitignore` entries:

```gitignore
.env
.venv/
__pycache__/
*.pyc
```

If the repository is intended for public GitHub hosting, generated indexes and large document collections should also be reviewed before committing.

---

# Troubleshooting

## Streamlit Command Not Found

Run:

```bash
python -m streamlit run docintel/app.py
```

---

## Missing Dependency

Run:

```bash
pip install -r docintel/requirements.txt
```

---

## API Key Error

Make sure `.env` contains:

```env
ANTHROPIC_API_KEY=your_api_key_here
```

and restart the application.

---

## Retrieval Index Error

Rebuild the index:

```bash
python -m docintel.retrieval.index
```

##

---

# Project Objective

The objective of this project is to build a document intelligence system capable of understanding heterogeneous documents rather than relying only on plain text extraction.

The system combines:

```text
Document Ingestion
        |
        v
OCR + Visual Processing
        |
        v
Document Chunking
        |
        v
Hybrid Retrieval
        |
        v
Evidence Selection
        |
        v
Question Answering
        |
        v
Citation Validation
        |
        v
Answer + Evidence
```

This architecture makes the system suitable for document-heavy use cases such as:

* Academic documents
* Examination papers
* Technical reports
* Business reports
* Research documents
* Scanned documents
* Tables and charts
* Knowledge-base question answering

---

# Future Improvements

Possible future enhancements include:

* Better multimodal embedding models
* More advanced vision-language models
* Improved table understanding
* Automatic chart data extraction
* Vector database integration
* MongoDB integration for document metadata
* User authentication
* Multi-user document collections
* Conversation history
* Streaming responses
* Improved citation highlighting
* Cloud deployment
* Docker support
* REST API support
* Automatic evaluation dashboards
* Advanced reranking models

---

# Project Information

**Project:** Multimodal Document Intelligence

**Technology Area:** Generative AI, Vision-Language Models, RAG, Document AI

**Primary Interface:** Streamlit

**Programming Language:** Python

**Architecture:** Document Ingestion, Retrieval, Evidence-Based Answering

---

# Summary

This project provides an end-to-end Multimodal Document Intelligence and RAG pipeline capable of processing text, scanned pages, tables, charts, and images.

Its key design principle is:

> Retrieve evidence first, then generate an answer grounded in that evidence with verifiable citations.
