"""
Chunking strategy (see DESIGN.md section 2 for the full rationale):

1. Walk the markdown line by line, tracking a *heading stack* so every
   chunk knows its section path, e.g.
   "Leave Policy > 4. Carry-forward > 4.1 Casual leave carry-forward".
2. Markdown tables are detected and chunked ROW-WISE, not as one blob.
   Each row becomes its own chunk, rendered as "<row header>: <col> is
   <value>" sentences, so a question about one cell ("does Standard
   cover dental implants?") retrieves a small, precise chunk instead of
   a whole table where the signal is diluted across many cells.
   A second, whole-table chunk is also stored, for questions that need
   to compare rows ("which tier has the highest LTA?").
3. Prose paragraphs are grouped up to CHUNK_TARGET_CHARS with a small
   trailing overlap, but never allowed to cross a heading boundary --
   a chunk always belongs to exactly one section.
"""
import re
from dataclasses import dataclass, field
from typing import List

from app.config import CHUNK_TARGET_CHARS, CHUNK_OVERLAP_CHARS

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
TABLE_ROW_RE = re.compile(r"^\|(.+)\|\s*$")
TABLE_SEP_RE = re.compile(r"^\|[\s:|-]+\|\s*$")


@dataclass
class Chunk:
    text: str
    section_path: str
    chunk_type: str  # "prose" | "table_row" | "table_full"


def _split_table_row(line: str) -> List[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _flush_table(rows: List[List[str]], section_path: str) -> List[Chunk]:
    """Turn a parsed markdown table into row-level + one full-table chunk."""
    if len(rows) < 2:
        return []
    header = rows[0]
    body_rows = rows[1:]

    chunks: List[Chunk] = []
    for row in body_rows:
        if not row or all(c == "" for c in row):
            continue
        row_label = row[0]
        pairs = []
        for col_name, val in zip(header[1:], row[1:]):
            pairs.append(f"{col_name} is {val}")
        sentence = f"For {header[0]} = {row_label}: " + "; ".join(pairs) + "."
        chunks.append(Chunk(text=sentence, section_path=section_path, chunk_type="table_row"))

    # Whole-table chunk (markdown re-rendered as plain text) for
    # cross-row / comparison questions.
    full_lines = [" | ".join(header)]
    for row in body_rows:
        full_lines.append(" | ".join(row))
    full_text = "Full table:\n" + "\n".join(full_lines)
    chunks.append(Chunk(text=full_text, section_path=section_path, chunk_type="table_full"))
    return chunks


def _flush_prose(buffer: List[str], section_path: str) -> List[Chunk]:
    text = "\n".join(buffer).strip()
    if not text:
        return []
    if len(text) <= CHUNK_TARGET_CHARS:
        return [Chunk(text=text, section_path=section_path, chunk_type="prose")]

    # Split on paragraph boundaries, accumulate up to target size with overlap.
    paras = [p for p in text.split("\n\n") if p.strip()]
    chunks: List[Chunk] = []
    current = ""
    for para in paras:
        if current and len(current) + len(para) + 2 > CHUNK_TARGET_CHARS:
            chunks.append(Chunk(text=current.strip(), section_path=section_path, chunk_type="prose"))
            # carry a small tail forward as overlap for continuity
            tail = current[-CHUNK_OVERLAP_CHARS:]
            current = tail + "\n\n" + para
        else:
            current = (current + "\n\n" + para) if current else para
    if current.strip():
        chunks.append(Chunk(text=current.strip(), section_path=section_path, chunk_type="prose"))
    return chunks


def chunk_markdown(doc_title: str, raw_text: str) -> List[Chunk]:
    lines = raw_text.splitlines()
    heading_stack: List[str] = [doc_title]
    chunks: List[Chunk] = []

    prose_buffer: List[str] = []
    table_buffer: List[List[str]] = []
    in_table = False

    def section_path() -> str:
        # Collapse consecutive duplicate titles (common when a doc's H1
        # repeats the filename-derived title) so citations read cleanly.
        deduped = []
        for title in heading_stack:
            if not deduped or deduped[-1].strip().lower() != title.strip().lower():
                deduped.append(title)
        return " > ".join(t for t in deduped if t)

    def flush_prose():
        nonlocal prose_buffer
        chunks.extend(_flush_prose(prose_buffer, section_path()))
        prose_buffer = []

    def flush_table():
        nonlocal table_buffer, in_table
        chunks.extend(_flush_table(table_buffer, section_path()))
        table_buffer = []
        in_table = False

    for line in lines:
        heading_match = HEADING_RE.match(line)
        if heading_match:
            flush_prose()
            if in_table:
                flush_table()
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            # heading_stack[0] is always the doc title. A level-1 "#" heading
            # sits at index 1, level-2 "##" at index 2, etc. Trim anything
            # deeper than the current heading, pad if we jumped a level,
            # then set this depth's title.
            heading_stack = heading_stack[:level]
            while len(heading_stack) < level:
                heading_stack.append("")
            heading_stack.append(title)
            continue

        if TABLE_SEP_RE.match(line):
            # the "| --- | --- |" separator row -- skip, keeps header/body split
            continue

        row_match = TABLE_ROW_RE.match(line)
        if row_match:
            if not in_table:
                flush_prose()
                in_table = True
            table_buffer.append(_split_table_row(line))
            continue

        # Not a table line -> close any open table first
        if in_table:
            flush_table()

        prose_buffer.append(line)

    flush_prose()
    if in_table:
        flush_table()

    return chunks
