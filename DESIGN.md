# DESIGN.md — HR Policy RAG Assistant

## 1. Architecture

```
                 ┌────────────────────┐
   admin ──upload─▶  POST /admin/documents │
                 └─────────┬──────────┘
                           │ .md/.txt/.pdf
                           ▼
                 ┌────────────────────┐
                 │  ingest.py         │
                 │  - read/decode     │
                 │  - chunk_markdown()│──▶ chunking.py (section+table aware)
                 │  - embed_texts()   │──▶ embeddings.py (sentence-transformers, local)
                 └─────────┬──────────┘
                           ▼
                 ┌────────────────────┐
                 │  SQLite (store.py) │  documents(id, filename, uploaded_at)
                 │                    │  chunks(id, doc_id, section_path,
                 │                    │         chunk_type, text, embedding JSON)
                 └─────────┬──────────┘
                           ▲
                           │ cosine search (numpy, brute force)
                           │
   employee ──ask──▶ POST /query ──▶ rag.py
                                       │
                                       ├─ embed question
                                       ├─ retrieve top-K chunks
                                       ├─ fast-refuse if best score < threshold
                                       ├─ build labeled context prompt
                                       ├─ call LLM (Groq/Gemini/Ollama) → strict JSON
                                       ├─ validate with Pydantic (LLMAnswer)
                                       ├─ ungrounded-answer guard
                                       └─ map chunk labels → Citation[]
                                       ▼
                              QueryResponse{answer, citations[], sufficient}
```

Components:
- **API**: FastAPI (`app/main.py`). Two real endpoints: `POST /admin/documents` (upload+index, admin-only via header) and `POST /query` (ask). Plus `GET /documents` and `GET /health`.
- **Chunker**: `app/chunking.py`, pure Python, no LLM involved.
- **Embeddings**: `app/embeddings.py`, local `sentence-transformers/all-MiniLM-L6-v2` — free, no API key, works offline after the first model download.
- **Store**: `app/store.py`, SQLite. Chunks and their embeddings live in one table; retrieval is a brute-force cosine scan in Python/numpy.
- **LLM**: `app/llm.py`, a 3-way swappable client (Groq default, Gemini, Ollama) behind one `complete_json()` function.
- **Orchestration/grounding**: `app/rag.py` — the anti-hallucination logic lives here, not in the prompt alone (see §3).
- **UI**: `ui/streamlit_app.py`, a single-page chat + admin upload panel that talks to the API over HTTP.

Data flow is synchronous end-to-end: upload blocks until chunked+embedded+stored; a query blocks until retrieval+LLM+validation complete. This is a deliberate simplification (see §5).

## 2. Chunking & retrieval

**Why markdown-aware, section-tracked chunking instead of fixed-size sliding windows:**
HR policies are short, dense, and *structured* — a wrong-section citation ("dental implants" answered from the LTA section) is worse than a slightly awkward chunk boundary. So chunking walks the document line-by-line and:

1. **Tracks a heading stack** so every chunk carries a full section path, e.g. `Leave Policy > 4. Carry-forward and encashment > 4.1 Casual leave carry-forward`. This becomes the citation's `section` field directly — no separate "find the nearest heading" step at query time.
2. **Never lets a chunk cross a heading boundary.** A paragraph that continues into the next section is still split there. This keeps citations honest: a chunk's section path is unambiguous.
3. **Chunks markdown tables row-wise**, not as one blob. Each row becomes a sentence like *"For Health tier = Standard: Eligible bands is Band B, Band C; Annual sum insured is ₹5,00,000; Dental implants is Not covered; ..."*. This is the single most important decision for the "table / structured policy" scenario in the assignment: embedding a whole 6-column table as one vector dilutes the signal for "does Standard cover dental implants?" across everything in the table. A row-level chunk is small, specific, and matches that question almost 1:1 in embedding space. A **whole-table chunk is also kept** (rendered as pipe-delimited text) for cross-row questions like "which tier has the highest LTA?", where the model needs to compare rows.
4. **Prose is paragraph-batched** up to `CHUNK_TARGET_CHARS` (default 700 chars) with a small trailing overlap (120 chars) carried into the next chunk, so a sentence split across a boundary isn't orphaned from its context.

**Metadata stored per chunk:** `filename`, `section_path`, `chunk_type` (`prose` / `table_row` / `table_full`), and the embedding vector. `chunk_type` isn't currently used to bias ranking, but is there for a cheap follow-up (e.g. boost `table_row` scores for questions matching a "does X cover Y" pattern — see §6).

**Retrieval:** top-K (default 5) by cosine similarity, computed with a brute-force numpy scan over all stored embeddings. See §5 for the scale trade-off.

## 3. Grounding — how hallucination is stopped

Grounding here is **prompt + retrieval + schema + a post-hoc guard working together**, not the prompt alone:

1. **Retrieval gate.** If the best-matching chunk's cosine similarity is below `MIN_SIMILARITY` (default 0.30), the LLM is **never called**. The question is refused immediately with the safe fallback. This handles clearly off-topic questions ("what's the weather") cheaply and makes off-policy refusal deterministic rather than dependent on the model's judgment.
2. **Context restriction.** The LLM only ever sees the retrieved chunks, each labeled `C1`, `C2`, ... with their source/section. It is never given the full document, so it cannot "fill in" from a different section it wasn't shown.
3. **Forced structured output + self-citation.** The system prompt requires strict JSON: `{"answer", "used_chunk_labels", "sufficient"}`. The model must name which labels it relied on. This does two things: it gives us machine-checkable citations (we map labels → real `Citation` objects after validation, so a citation can never point to a document/section the model invented), and it forces the model to "show its work," which empirically reduces confident-but-ungrounded answers.
4. **Explicit refusal instruction.** The prompt states plainly: if the chunks don't answer the question, set `sufficient=false` and point to HR — do not guess.
5. **Ungrounded-answer guard (post-validation).** Even with all of the above, a model could still return `sufficient=true` with an empty `used_chunk_labels` (a hallucinated "yes" with no support). `rag.py` explicitly checks for this and forces a refusal if it happens. This is the one check that isn't "trust the prompt" — it's a hard rule in code.
6. **Retry-then-refuse on malformed output.** If the LLM's JSON fails to parse or fails Pydantic validation, we retry once with a stricter reminder; if it still fails, we return the safe fallback rather than a 500 error or a raw/garbled answer.

**What happens when retrieval is weak but not zero** (e.g. the question is adjacent to a policy but not answered by it, like "how long can I keep my laptop unencrypted?" against a policy that only says encryption is required): retrieval still returns *some* chunk (probably the encryption sentence), scoring above the fast-refusal threshold. This is exactly why step 3–5 exist — the LLM is instructed to recognize that the retrieved chunk states a requirement but doesn't answer "how long," and must say so via `sufficient=false`, not confidently invent a grace period. The eval set (`eval/eval_set.json`) includes this exact case (`off_policy_tricky`) to catch regressions here.

## 4. Schema & APIs

**`POST /admin/documents`** (multipart file upload, requires header `X-Role: admin`)
```json
{ "document_id": 4, "filename": "benefits-policy.md", "num_chunks": 18 }
```
Why a header instead of a body flag: it mirrors how a real auth layer would inject role information (e.g. from a JWT claim) without building real auth, per the assignment's explicit allowance for a hardcoded flag.

**`POST /query`**
Request:
```json
{ "question": "Does the Standard health tier cover dental implants?" }
```
Response:
```json
{
  "answer": "The Standard health tier does not cover dental implants.",
  "citations": [
    { "document": "benefits-policy.md", "section": "Benefits Policy > 2. Health coverage tiers", "chunk_id": 7 }
  ],
  "sufficient": true,
  "retrieved_chunk_ids": [7, 12, 3, 9, 14]
}
```
On refusal, `citations` is `[]`, `sufficient` is `false`, and `answer` is the safe fallback string. `retrieved_chunk_ids` is included even on refusal so a reviewer/eval script can see *what was retrieved* even when the model declined to answer — useful for debugging whether a refusal was a retrieval gap or a grounding decision.

Why this shape: the assignment requires "structured data (JSON with answer + citations[])," and a Pydantic model (`QueryResponse`) enforces that shape server-side, not just informally.

**`GET /documents`** — list indexed documents with chunk counts, for the admin panel and for sanity-checking ingestion.

## 5. Trade-offs

1. **Brute-force cosine search (numpy) vs. a vector DB (Chroma/pgvector).**
   Chosen: brute force over all embeddings in SQLite.
   Rejected: Chroma/pgvector.
   Why: at HR-policy scale (a handful of documents, low hundreds to low thousands of chunks), a full scan is single-digit milliseconds and removes an entire service dependency, which matters for a 10-minute local setup. The trade-off is real, though — this does not scale past roughly tens of thousands of chunks, and there's no approximate-nearest-neighbor index. If the document set grew to company-wide, multi-department scale, pgvector (keeps the "one SQL store" property) would be the first upgrade, not a bespoke ANN implementation.

2. **Synchronous ingestion vs. async/background ingestion.**
   Chosen: the upload endpoint blocks until chunking + embedding + storage finish.
   Rejected: a job queue (e.g. Celery/RQ) with a "processing" status.
   Why: policy documents are small (a few KB to a few hundred KB) and embedding a full document locally takes well under a second on CPU. Async ingestion is genuinely the right call for large batch uploads or very large PDFs, but adding a queue and worker process for documents this size would be complexity without a corresponding benefit, and the assignment explicitly scopes out infra hardening.

3. **Table handling: row-level + whole-table chunks (redundant storage) vs. table-as-one-chunk only.**
   Chosen: store both a per-row chunk and a whole-table chunk.
   Rejected: one chunk per table.
   Why: a single "does X cover Y" question needs a small, precise chunk; a comparison question needs the whole table. Storing both roughly doubles the chunk count for tables (which are small anyway — a few rows) in exchange for handling both question shapes well. The alternative (table-as-one-chunk only) would have made the assignment's own "table / structured policy" example scenario noticeably less precise.

4. **LLM provider: swappable (Groq default) vs. hard-committing to one vendor.**
   Chosen: a thin `complete_json()` abstraction over Groq/Gemini/Ollama.
   Rejected: hard-coding one provider's SDK throughout the codebase.
   Why: free-tier quotas for any single provider change quickly (the assignment calls this out explicitly), so the grounding/retrieval logic — the actual point of the exercise — shouldn't be coupled to one vendor's client library. The cost is a slightly thinner integration with each provider (e.g. no provider-specific function-calling), which is an acceptable trade for a take-home.

5. **Character-based chunk sizing vs. token-based.**
   Chosen: `CHUNK_TARGET_CHARS` (characters).
   Rejected: a tokenizer-based length function.
   Why: avoids pulling in a tokenizer dependency (tiktoken/etc.) purely for chunk sizing, when policy prose is short and consistent enough that character count is a good-enough proxy for token count. Would need to revisit this if the assistant ever ingested very technical or non-English text where the char:token ratio varies more.

## 6. If I had two more weeks

Roughly in priority order:

1. **A real (bigger) eval harness.** The current `eval/eval_set.json` has 7 hand-picked cases. I'd grow this to 30–50 cases per document, including paraphrased questions ("how much sick leave do I get" vs. "sick leave entitlement"), and track pass rate over time as a regression gate before merging prompt/chunking changes.
2. **Hybrid search (keyword + vector).** Clause numbers, exact figures, and named policy sections (e.g. "section 4.1") are exactly the kind of query embeddings handle worst. Adding a keyword/BM25 pass (even SQLite FTS5) alongside the vector search, then merging results, would materially help these.
3. **Confidence-aware refusal, not just a similarity cutoff.** Right now the fast-refusal threshold is a single tuned constant. I'd want to log real query/similarity pairs from usage and calibrate this against actual precision/recall, and possibly make it category-aware (a "which tier" comparison question might legitimately need a lower per-chunk similarity than a "what is X" lookup).
4. **PDF table extraction.** Current PDF support does raw text extraction (`pypdf`), which loses table structure — a PDF with the same benefits table would not get the row-level chunking benefit described in §2. I'd add a table-detection library (e.g. `pdfplumber`'s table extraction) and route detected tables through the same row-chunking logic as markdown.
5. **Versioning / re-indexing on replace.** Today, re-uploading a file with the same name deletes and replaces its chunks (crude versioning). I'd add an explicit version history so old citations don't silently point to language that no longer exists, and so "what changed in the last policy update" becomes answerable.
6. **Thumbs up/down feedback**, stored per query+answer, to build a real signal for which retrieval/prompt changes actually help rather than guessing from the eval set alone.
7. **Real auth**, replacing the hardcoded `X-Role` header with actual session-based admin/employee roles, once there's more than one admin.
