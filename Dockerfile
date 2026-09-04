FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY seed_data/ seed_data/
COPY seed.py .

# Bake the sample policies into the image so a fresh deploy is queryable
# immediately. Re-uploading via /admin/documents still works at runtime.
RUN python seed.py

EXPOSE 8000
# Render (and similar PaaS) inject their own $PORT and expect the app to
# bind to it -- a hardcoded port can cause "no open ports detected" even
# though the app started fine internally. Shell form so $PORT expands;
# falls back to 8000 for local `docker run` / docker-compose.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
