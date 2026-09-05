from pathlib import Path

from app.chunking import chunk_markdown
from app.embeddings import embed_texts
from app import store

def ingest_text_document(filename: str, raw_text: str, replace_existing: bool = True) -> dict:
    if replace_existing:
        store.delete_document_by_filename(filename)

    doc_title = Path(filename).stem.replace("-", " ").replace("_", " ").title()
    chunks = chunk_markdown(doc_title, raw_text)
    if not chunks:
        raise ValueError("No content could be extracted from this file.")

    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)

    doc_id = store.add_document(filename)
    rows = [
        (c.section_path, c.chunk_type, c.text, emb)
        for c, emb in zip(chunks, embeddings)
    ]
    store.add_chunks(doc_id, filename, rows)

    return {"document_id": doc_id, "filename": filename, "num_chunks": len(chunks)}


def ingest_pdf_document(filename: str, file_bytes: bytes, replace_existing: bool = True) -> dict:
    """Best-effort PDF text extraction (stretch: PDF support)."""
    try:
        from pypdf import PdfReader
        import io
        reader = PdfReader(io.BytesIO(file_bytes))
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        raise ValueError(f"Could not read PDF: {e}")

    if not text.strip():
        raise ValueError("PDF appears to have no extractable text (likely scanned/image-only).")

    # PDFs rarely have clean markdown headings, so we treat each
    # double-newline block as its own pseudo-section, prefixed with the
    # doc title, rather than pretending we found a heading structure.
    doc_title = Path(filename).stem.replace("-", " ").replace("_", " ").title()
    pseudo_markdown = f"# {doc_title}\n\n" + text
    return ingest_text_document(filename, pseudo_markdown, replace_existing=replace_existing)
