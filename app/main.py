from typing import List

from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app import store
from app.config import ADMIN_HEADER, ADMIN_VALUE
from app.ingest import ingest_text_document, ingest_pdf_document
from app.rag import answer_question
from app.schemas import QueryRequest, QueryResponse, DocumentInfo

app = FastAPI(title="HR Policy RAG Assistant", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    store.init_db()


def _require_admin(role: str | None):
    # Hardcoded role flag, exactly as the assignment allows ("Out of
    # scope: real SSO / RBAC"). A header is enough to demonstrate the
    # admin-vs-employee boundary without building real auth.
    if role != ADMIN_VALUE:
        raise HTTPException(status_code=403, detail=f"Admin access required. Send header '{ADMIN_HEADER}: {ADMIN_VALUE}'.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents", response_model=List[DocumentInfo])
def list_documents():
    return store.list_documents()


@app.post("/admin/documents")
async def upload_document(file: UploadFile = File(...), x_role: str | None = Header(default=None)):
    _require_admin(x_role)

    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    lower = filename.lower()
    try:
        if lower.endswith((".md", ".txt", ".markdown")):
            text = content.decode("utf-8", errors="replace")
            result = ingest_text_document(filename, text)
        elif lower.endswith(".pdf"):
            result = ingest_pdf_document(filename, content)
        else:
            raise HTTPException(status_code=400, detail="Only .md, .txt, and .pdf files are supported.")
    except ValueError as e:
        # Unreadable/garbage file -> 422, not a 500 crash.
        raise HTTPException(status_code=422, detail=str(e))

    return result


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    return answer_question(req.question)
