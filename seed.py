"""Load the sample policies in seed_data/ into the index. Run once after
`uvicorn app.main:app` is up, or standalone (it talks to the DB directly,
no server needed): `python seed.py`
"""
from pathlib import Path

from app import store
from app.ingest import ingest_text_document

SEED_DIR = Path(__file__).resolve().parent / "seed_data"


def main():
    store.init_db()
    for path in sorted(SEED_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        result = ingest_text_document(path.name, text)
        print(f"Indexed {result['filename']}: {result['num_chunks']} chunks")


if __name__ == "__main__":
    main()
