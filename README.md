# HR Policy RAG Assistant

A small Retrieval-Augmented-Generation service for HR policy Q&A. Employees ask
questions in plain English; answers come **only** from uploaded policy
documents, with citations, and the system refuses when the policies don't
say — no guessing. See [`DESIGN.md`](./DESIGN.md) for the full architecture
and reasoning.

## Stack

- **API**: FastAPI (Python)
- **Embeddings**: local, free — `fastembed` (pure ONNX Runtime, no PyTorch) running `sentence-transformers/all-MiniLM-L6-v2` (no API key, no quota, small memory footprint for constrained hosting)
- **LLM**: swappable — **Groq** (default, free tier, fast) / Gemini / Ollama (fully local, no key)
- **Store**: SQLite (documents + chunks + embeddings), brute-force cosine search
- **UI**: Streamlit

## 1. Local setup (under 10 minutes)

```bash
git clone <this-repo>
cd hr-rag-assistant

python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
```

### Get a free LLM key (pick one)

- **Groq (default, recommended)** — https://console.groq.com → create a free API key → paste into `.env` as `GROQ_API_KEY`. No credit card.
- **Gemini** — https://aistudio.google.com/apikey → free tier → set `.env`: `LLM_PROVIDER=gemini`, `GEMINI_API_KEY=...`.
- **Ollama (fully local, no key)** — install [Ollama](https://ollama.com), run `ollama pull llama3.1 && ollama serve`, set `.env`: `LLM_PROVIDER=ollama`.

Embeddings never need a key — the first run downloads the `all-MiniLM-L6-v2` ONNX weights (~80MB) via `fastembed` once and caches them locally.

## 2. Seed the sample policies and start the API

```bash
python seed.py                              # indexes seed_data/*.md
uvicorn app.main:app --reload --port 8000
```

Check it's alive: `curl http://localhost:8000/health` → `{"status":"ok","message":"app pong"}`
(`/ping` is aliased to the same handler, for platforms that health-check that path by convention.)

## 3. Run the UI

In a second terminal (same venv):
```bash
streamlit run ui/streamlit_app.py
```
Open the URL Streamlit prints (usually http://localhost:8501). Check "I'm an admin" in the sidebar to upload additional `.md` / `.txt` / `.pdf` policies.

## 4. Or use the API directly (CLI-friendly)

Upload a new policy (admin):
```bash
curl -X POST http://localhost:8000/admin/documents \
  -H "X-Role: admin" \
  -F "file=@seed_data/leave-policy.md"
```

Ask a question:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the casual leave carry-forward limit?"}'
```

Example response:
```json
{
  "answer": "Up to 8 days of unused casual leave may be carried forward to the next calendar year, and must be used by 31 March of that year.",
  "citations": [
    { "document": "leave-policy.md", "section": "Leave Policy > 4. Carry-forward and encashment > 4.1 Casual leave carry-forward", "chunk_id": 12 }
  ],
  "sufficient": true,
  "retrieved_chunk_ids": [12, 3, 8, 1, 15]
}
```

Off-policy question:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Can I expense a personal home gym?"}'
```
```json
{
  "answer": "I don't have enough information in the uploaded policies to answer that. Please contact HR.",
  "citations": [],
  "sufficient": false,
  "retrieved_chunk_ids": [4, 9, 2, 11, 6]
}
```

## 5. Run with Docker

Two containers are provided — one for the API, one for the Streamlit UI:

```bash
cp .env.example .env    # fill in GROQ_API_KEY (or switch provider)
docker compose up --build
```

- API: http://localhost:8000
- UI: http://localhost:8501 (talks to the API container at `http://api:8000`)

Notes:
- `Dockerfile` runs `python seed.py` at **build time**, so a fresh image is queryable immediately with the sample policies already indexed. Uploading via `/admin/documents` at runtime still works and adds to the same index.
- `./data` is mounted as a volume into the API container, so the SQLite DB and any uploaded raw docs persist across `docker compose down`/`up`.
- The API container's `CMD` binds to `$PORT` if set (falling back to `8000`), so the same image deploys cleanly on PaaS platforms (e.g. Render) that inject their own port — a hardcoded port there can cause a "no open ports detected" failure even though the app is running fine internally.

## 6. Run the tests / eval

Unit tests (chunking logic, no server or API key needed):
```bash
python -m unittest discover -s tests -v
```

Tiny eval set (needs the API server running and seeded):
```bash
python eval/run_eval.py
```

## Environment variables

See [`.env.example`](./.env.example) for the full list. Key ones:

| Variable | Purpose | Default |
| --- | --- | --- |
| `LLM_PROVIDER` | `groq` \| `gemini` \| `ollama` | `groq` |
| `GROQ_API_KEY` | Groq free-tier key | — |
| `EMBEDDING_MODEL` | Local embedding model, resolved through `fastembed` | `sentence-transformers/all-MiniLM-L6-v2` |
| `TOP_K` | Chunks retrieved per query | `5` |
| `MIN_SIMILARITY` | Refuse before calling the LLM below this cosine score | `0.30` |
| `CHUNK_TARGET_CHARS` | Target prose chunk size | `700` |
| `DB_PATH` | SQLite file location | `data/hr_rag.sqlite3` |
| `UPLOAD_DIR` | Where raw uploaded docs are kept | `data/raw_docs` |
| `PORT` | (Docker/Render only) port the API binds to | `8000` |

## Notes

- **A hardcoded admin flag** (`X-Role: admin` header) stands in for real auth, as explicitly allowed by the assignment scope.
- **No API keys are committed.** `.env` is git-ignored; `.env.example` documents every variable.
- If you're outside a region where Google's free tier reuses prompts to improve products, that applies only if you choose the Gemini path — Groq and Ollama don't have that caveat.