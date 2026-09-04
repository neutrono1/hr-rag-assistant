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
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
