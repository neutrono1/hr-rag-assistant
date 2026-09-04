"""
Central configuration, loaded from environment variables (.env).
Keeping this in one place makes the "why" of each default easy to audit.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = os.getenv("DB_PATH", str(BASE_DIR / "data" / "hr_rag.sqlite3"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(BASE_DIR / "data" / "raw_docs")))

# --- Embeddings (always local / free) ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --- LLM provider: "groq" | "gemini" | "ollama" ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

# --- Retrieval / grounding knobs ---
TOP_K = int(os.getenv("TOP_K", "5"))
# Cosine similarity below this -> refuse before even calling the LLM.
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.30"))
# Chunking targets (characters, not tokens, to keep the dependency list small)
CHUNK_TARGET_CHARS = int(os.getenv("CHUNK_TARGET_CHARS", "700"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "120"))

# --- Fake auth (assignment explicitly allows this) ---
ADMIN_HEADER = "X-Role"
ADMIN_VALUE = "admin"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
