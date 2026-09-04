# HR Policy RAG Assistant

A small Retrieval-Augmented-Generation service for HR policy Q&A. Employees ask
questions in plain English; answers come **only** from uploaded policy
documents, with citations, and the system refuses when the policies don't
say — no guessing. See [`DESIGN.md`](./DESIGN.md) for the full architecture
and reasoning.

## Stack

- **API**: FastAPI (Python)
- **Embeddings**: local, free — `sentence-transformers/all-MiniLM-L6-v2` (no API key, no quota)
- **LLM**: swappable — **Groq** (default, free tier, fast) / Gemini / Ollama (fully local, no key)
- **Store**: SQLite (documents + chunks + embeddings), brute-force cosine search
- **UI**: Streamlit

## 1. Setup (under 10 minutes)

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

Embeddings never need a key — the first run downloads `all-MiniLM-L6-v2` (~80MB) once and caches it locally.

## 2. Seed the sample policies and start the API

```bash
python seed.py                              # indexes seed_data/*.md
uvicorn app.main:app --reload --port 8000
```

Check it's alive: `curl http://localhost:8000/health` → `{"status":"ok"}`

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

## 5. Run the tests / eval

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
| `EMBEDDING_MODEL` | Local sentence-transformers model | `all-MiniLM-L6-v2` |
| `TOP_K` | Chunks retrieved per query | `5` |
| `MIN_SIMILARITY` | Refuse before calling the LLM below this cosine score | `0.30` |
| `CHUNK_TARGET_CHARS` | Target prose chunk size | `700` |

## Notes

- **A hardcoded admin flag** (`X-Role: admin` header) stands in for real auth, as explicitly allowed by the assignment scope.
- **No API keys are committed.** `.env` is git-ignored; `.env.example` documents every variable.
- If you're outside a region where Google's free tier reuses prompts to improve products, that applies only if you choose the Gemini path — Groq and Ollama don't have that caveat.
