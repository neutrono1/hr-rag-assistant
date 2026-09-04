"""
Grounding strategy (see DESIGN.md section 3 for full write-up):

  1. Retrieve top-K chunks by cosine similarity.
  2. FAST REFUSAL: if the best similarity score is below MIN_SIMILARITY,
     we never call the LLM at all -- there's nothing relevant enough to
     answer from, so we return the safe fallback immediately. This also
     saves API quota on clearly off-topic questions.
  3. If retrieval looks plausible, we still don't trust the model by
     default. The prompt:
       - Gives the model ONLY the retrieved chunks (never the full doc).
       - Labels each chunk (C1, C2, ...) and requires the model to name
         which labels it actually used.
       - Requires strict JSON output validated against LLMAnswer.
       - Explicitly instructs: if the chunks don't answer the question,
         set sufficient=false and say so -- do not guess.
  4. Post-validation: if the model claims sufficient=true but returns
     zero used_chunk_labels, we treat that as an ungrounded answer and
     force a refusal (a model can still hallucinate a "yes" without
     pointing at any source; this catches that case).
  5. If the JSON fails to parse/validate, we retry once with a stricter
     reminder, then fall back to a safe refusal rather than erroring.
"""
from typing import List

from pydantic import ValidationError

from app.config import TOP_K, MIN_SIMILARITY
from app.embeddings import embed_one
from app.schemas import QueryResponse, Citation, LLMAnswer
from app.llm import complete_json, LLMError
from app import store

FALLBACK_MESSAGE = "I don't have enough information in the uploaded policies to answer that. Please contact HR."

SYSTEM_PROMPT = """You are an HR policy assistant. You answer ONLY using the numbered \
context chunks provided in the user message. Rules:

- Do not use outside knowledge, even if you are confident it is correct.
- Every claim in your answer must be traceable to at least one chunk.
- If the chunks do not contain enough information to answer confidently, \
set "sufficient" to false and write a short refusal in "answer" pointing \
the user to HR. Do not guess or infer beyond what the chunks state.
- Respond with ONLY a JSON object matching exactly this shape, no prose \
outside the JSON, no markdown fences:
{"answer": "<string>", "used_chunk_labels": ["C1", "C3"], "sufficient": true}
- "used_chunk_labels" must list only labels of chunks you actually relied on. \
If sufficient is false, used_chunk_labels should be an empty list.
"""


def _build_user_prompt(question: str, retrieved: List[dict]) -> str:
    lines = [f"Question: {question}", "", "Context chunks:"]
    for i, item in enumerate(retrieved, start=1):
        chunk = item["chunk"]
        label = f"C{i}"
        lines.append(f"[{label}] (source: {chunk['filename']} | section: {chunk['section_path']})")
        lines.append(chunk["text"])
        lines.append("")
    return "\n".join(lines)


def answer_question(question: str) -> QueryResponse:
    question = (question or "").strip()
    if not question:
        return QueryResponse(answer="Please enter a question.", citations=[], sufficient=False)

    q_embedding = embed_one(question)
    hits = store.search(q_embedding, top_k=TOP_K)

    if not hits or hits[0][1] < MIN_SIMILARITY:
        return QueryResponse(answer=FALLBACK_MESSAGE, citations=[], sufficient=False)

    retrieved = [{"chunk": chunk, "score": score} for chunk, score in hits]
    label_to_chunk = {f"C{i}": item["chunk"] for i, item in enumerate(retrieved, start=1)}
    user_prompt = _build_user_prompt(question, retrieved)

    llm_answer = _call_llm_with_retry(user_prompt)

    if llm_answer is None:
        return QueryResponse(
            answer=FALLBACK_MESSAGE,
            citations=[],
            sufficient=False,
            retrieved_chunk_ids=[c["id"] for c in label_to_chunk.values()],
        )

    # Ungrounded "yes" guard: sufficient=true but no chunks cited -> refuse.
    if llm_answer.sufficient and not llm_answer.used_chunk_labels:
        return QueryResponse(
            answer=FALLBACK_MESSAGE,
            citations=[],
            sufficient=False,
            retrieved_chunk_ids=[c["id"] for c in label_to_chunk.values()],
        )

    if not llm_answer.sufficient:
        return QueryResponse(
            answer=llm_answer.answer or FALLBACK_MESSAGE,
            citations=[],
            sufficient=False,
            retrieved_chunk_ids=[c["id"] for c in label_to_chunk.values()],
        )

    citations = []
    for label in llm_answer.used_chunk_labels:
        chunk = label_to_chunk.get(label)
        if chunk:
            citations.append(Citation(document=chunk["filename"], section=chunk["section_path"], chunk_id=chunk["id"]))

    return QueryResponse(
        answer=llm_answer.answer,
        citations=citations,
        sufficient=True,
        retrieved_chunk_ids=[c["id"] for c in label_to_chunk.values()],
    )


def _call_llm_with_retry(user_prompt: str) -> LLMAnswer:
    for attempt in range(2):
        try:
            prompt = user_prompt
            if attempt == 1:
                prompt += "\n\nReminder: reply with ONLY the JSON object, matching the exact schema."
            raw = complete_json(SYSTEM_PROMPT, prompt)
            return LLMAnswer(**raw)
        except (LLMError, ValidationError, ValueError, KeyError, TypeError):
            continue
    return None
