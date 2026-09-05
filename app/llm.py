"""
Thin, swappable LLM client. All providers expose one function:
    complete_json(system_prompt, user_prompt) -> str (raw JSON text)

We ask every provider for JSON-mode / JSON-biased output, then validate
with Pydantic in rag.py. If parsing fails we retry once with a stricter
reminder before falling back to a safe refusal.
"""
import json
import httpx

from app.config import (
    LLM_PROVIDER, GROQ_API_KEY, GROQ_MODEL,
    GEMINI_API_KEY, GEMINI_MODEL, OLLAMA_HOST, OLLAMA_MODEL,
)


class LLMError(Exception):
    pass


def _groq_complete(system_prompt: str, user_prompt: str) -> str:
    if not GROQ_API_KEY:
        raise LLMError("GROQ_API_KEY is not set. See .env.example.")
    resp = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _gemini_complete(system_prompt: str, user_prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise LLMError("GEMINI_API_KEY is not set. See .env.example.")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    try:
        resp = httpx.post(
            url,
            json={
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "generationConfig": {
                    "temperature": 0,
                    "responseMimeType": "application/json",
                },
            },
            timeout=30,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise LLMError(f"Gemini API error: {e.response.status_code} {e.response.text[:200]}")
    except httpx.RequestError as e:
        raise LLMError(f"Gemini request failed: {e}")
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]

def _ollama_complete(system_prompt: str, user_prompt: str) -> str:
    resp = httpx.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]


_PROVIDERS = {
    "groq": _groq_complete,
    "gemini": _gemini_complete,
    "ollama": _ollama_complete,
}


def complete_json(system_prompt: str, user_prompt: str) -> dict:
    fn = _PROVIDERS.get(LLM_PROVIDER)
    if fn is None:
        raise LLMError(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Choose groq | gemini | ollama.")
    raw = fn(system_prompt, user_prompt)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)
