"""
Storage layer. SQLite holds documents, chunks, and embeddings (as JSON
arrays of floats). Similarity search is done in plain Python with numpy
cosine similarity rather than a dedicated vector DB.

Why: at HR-policy scale (dozens of docs, low thousands of chunks) a
brute-force scan is a few milliseconds and avoids an extra moving part.
See DESIGN.md section 5 for the trade-off vs. pgvector/Chroma at scale.
"""
import json
import sqlite3
import datetime
from typing import List, Optional, Tuple

import numpy as np

from app.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    uploaded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    filename TEXT NOT NULL,
    section_path TEXT NOT NULL,
    chunk_type TEXT NOT NULL,
    text TEXT NOT NULL,
    embedding TEXT NOT NULL,
    FOREIGN KEY(document_id) REFERENCES documents(id)
);
"""


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    conn = get_conn()
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def add_document(filename: str) -> int:
    conn = get_conn()
    try:
        cur = conn.execute(
            "INSERT INTO documents (filename, uploaded_at) VALUES (?, ?)",
            (filename, datetime.datetime.utcnow().isoformat()),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def delete_document_by_filename(filename: str) -> None:
    """Used when re-uploading a policy with the same name (simple versioning)."""
    conn = get_conn()
    try:
        rows = conn.execute("SELECT id FROM documents WHERE filename = ?", (filename,)).fetchall()
        for (doc_id,) in rows:
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
    finally:
        conn.close()


def add_chunks(document_id: int, filename: str, chunk_rows: List[Tuple[str, str, str, List[float]]]) -> None:
    """chunk_rows: list of (section_path, chunk_type, text, embedding)"""
    conn = get_conn()
    try:
        conn.executemany(
            "INSERT INTO chunks (document_id, filename, section_path, chunk_type, text, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (document_id, filename, section_path, chunk_type, text, json.dumps(embedding))
                for section_path, chunk_type, text, embedding in chunk_rows
            ],
        )
        conn.commit()
    finally:
        conn.close()


def list_documents() -> List[dict]:
    conn = get_conn()
    try:
        docs = conn.execute("SELECT id, filename, uploaded_at FROM documents ORDER BY id").fetchall()
        out = []
        for doc_id, filename, uploaded_at in docs:
            (count,) = conn.execute("SELECT COUNT(*) FROM chunks WHERE document_id = ?", (doc_id,)).fetchone()
            out.append({"id": doc_id, "filename": filename, "uploaded_at": uploaded_at, "num_chunks": count})
        return out
    finally:
        conn.close()


def all_chunks_with_embeddings() -> List[dict]:
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT id, filename, section_path, chunk_type, text, embedding FROM chunks"
        ).fetchall()
        return [
            {
                "id": r[0],
                "filename": r[1],
                "section_path": r[2],
                "chunk_type": r[3],
                "text": r[4],
                "embedding": json.loads(r[5]),
            }
            for r in rows
        ]
    finally:
        conn.close()


def search(query_embedding: List[float], top_k: int) -> List[Tuple[dict, float]]:
    """Brute-force cosine similarity search across all stored chunks."""
    chunks = all_chunks_with_embeddings()
    if not chunks:
        return []
    q = np.array(query_embedding, dtype=np.float32)
    q_norm = np.linalg.norm(q) + 1e-8

    scored = []
    for c in chunks:
        v = np.array(c["embedding"], dtype=np.float32)
        v_norm = np.linalg.norm(v) + 1e-8
        sim = float(np.dot(q, v) / (q_norm * v_norm))
        scored.append((c, sim))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]
