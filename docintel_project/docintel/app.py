"""Streamlit UI with Automatic Document Ingestion and Question Answering."""
import os
import sys

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st

from docintel.answering.answer import MissingAPIKeyError, answer_with_context
from docintel.ingestion.uploader import (
    save_and_index_file,
    list_indexed_documents,
    sync_all_docs_in_directory,
    delete_indexed_document
)

st.set_page_config(page_title="Document QA & Intelligence", layout="wide")

# Ensure all files on disk are automatically synchronized and indexed
if "synced_on_startup" not in st.session_state:
    try:
        newly_synced = sync_all_docs_in_directory()
        st.session_state["synced_on_startup"] = True
    except Exception:
        pass

# ------------------------------------------------------------- Sidebar: Upload & Docs
with st.sidebar:
    st.header("📂 Document Management")
    st.caption("Upload PDFs or text files to add them to the knowledge base.")

    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Upload PDF or text documents to index them automatically."
    )

    if uploaded_files:
        if "processed_files" not in st.session_state:
            st.session_state["processed_files"] = set()

        new_files = [f for f in uploaded_files if f.name not in st.session_state["processed_files"]]
        if new_files:
            with st.spinner("Processing and indexing uploaded documents..."):
                for uf in new_files:
                    try:
                        pages, chunks = save_and_index_file(uf.name, uf.getvalue())
                        st.session_state["processed_files"].add(uf.name)
                        st.success(f"✅ Indexed **{uf.name}** ({pages} pages, {chunks} chunks)")
                    except Exception as e:
                        st.error(f"Failed to index {uf.name}: {e}")
                st.rerun()

    st.divider()
    st.subheader("📚 Indexed Documents")
    indexed_docs = list_indexed_documents()
    if indexed_docs:
        for doc in indexed_docs:
            c1, c2 = st.columns([0.82, 0.18])
            c1.markdown(f"• `{doc}`")
            if c2.button("🗑️", key=f"del_{doc}", help=f"Delete {doc}"):
                delete_indexed_document(doc)
                st.rerun()
    else:
        st.info("No documents indexed yet.")

    st.divider()
    with st.expander("💡 Example Questions"):
        st.markdown(
            "**For Software Engineering (Scanned Test Paper):**\n"
            "- *Explain about the full topic in the software engineering test document*\n"
            "- *What are the phases of SDLC in software engineering?*\n"
            "- *What are umbrella activities and what are their key tasks?*\n"
            "- *What is the waterfall model and what are its advantages and disadvantages?*\n"
            "- *Who is Siddharth and what is his register number?*\n\n"
            "**For Practicum 4:**\n"
            "- *Explain about the full topic in Practicum 4*\n"
            "- *What is the practical significance of cumulative distribution evaluation?*\n"
            "- *What is the formula and definition of CDF?*\n\n"
            "**For Plant Reports:**\n"
            "- *What are the three reasons given for the decline in efficiency from Q2 to Q4?*\n"
            "- *What is the percentage increase in mechanical failure downtime from Q2 to Q4?*"
        )

# ------------------------------------------------------------- Main Panel: QA
st.title("Document Question Answering")
st.caption("Answers are grounded in your indexed documents; every claim carries a citation.")

question = st.text_input(
    "Ask a question about your documents",
    placeholder="e.g. Explain about the full topic in Practicum 4"
)
go = st.button("Ask", type="primary")

if go and question.strip():
    try:
        with st.spinner("Retrieving evidence and reasoning..."):
            ans, chunks = answer_with_context(question.strip())
    except MissingAPIKeyError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Could not answer: {e}")
        st.stop()

    if not ans.supported:
        st.warning("Not supported by documents")
        st.write(ans.answer)
    else:
        st.subheader("Answer")
        st.markdown(ans.answer)

    if ans.citations:
        st.subheader("Citations")
        st.dataframe(
            pd.DataFrame([
                {
                    "Document": c.doc_id,
                    "Page": c.page,
                    "Section": c.section,
                    "Quote / value": c.quote_or_value
                } for c in ans.citations
            ]),
            use_container_width=True,
            hide_index=True
        )

    if ans.calculations:
        st.subheader("Calculation steps")
        for line in ans.calculations:
            st.code(line, language=None)

    # Cited page images, side by side (if available)
    pages, seen = [], set()
    for c in ans.citations:
        key = (c.doc_id, c.page)
        if key in seen:
            continue
        seen.add(key)
        ch = next((x for x in chunks if x.doc_id == c.doc_id and x.page == c.page and x.page_image), None)
        if ch and os.path.isfile(ch.page_image):
            pages.append((c, ch.page_image))
    if pages:
        st.subheader("Cited pages")
        cols = st.columns(min(len(pages), 3))
        for i, (c, path) in enumerate(pages):
            cols[i % len(cols)].image(path, caption=f"{c.doc_id} - page {c.page}", use_container_width=True)

elif go:
    st.info("Please enter a question.")
