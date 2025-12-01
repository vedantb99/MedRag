import json
import os
import re
from typing import Any, Dict, List

import requests
import torch
from sentence_transformers import util
from transformers import AutoModel, AutoTokenizer

from mara_pipelines.state import MaraGraphState

VLLM_SVA_ENDPOINT = os.getenv(
    "MEDRAG_SVA_ENDPOINT",
    "http://127.0.0.1:8000/v1/chat/completions",
)
VLLM_SVA_MODEL = os.getenv("MEDRAG_SVA_MODEL", "google/medgemma-4b-it")
SIMILARITY_THRESHOLD = float(os.getenv("MEDRAG_SVA_SIM_THRESHOLD", "0.7"))
# Use CPU for safety validation model to save GPU memory for vLLM and Vision models
_DEVICE = os.getenv("MEDRAG_SVA_DEVICE", "cpu")

_SVA_TOKENIZER = None
_SVA_MODEL = None


def _load_sva_encoder():
    global _SVA_TOKENIZER, _SVA_MODEL
    if _SVA_MODEL is None:
        print("SVA: loading Bio_ClinicalBERT encoder...")
        _SVA_TOKENIZER = AutoTokenizer.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT"
        )
        _SVA_MODEL = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        _SVA_MODEL.to(_DEVICE)
        _SVA_MODEL.eval()
    return _SVA_TOKENIZER, _SVA_MODEL


def _get_embeddings(texts: List[str]) -> torch.Tensor:
    tokenizer, model = _load_sva_encoder()
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
    ).to(_DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # mean pooling


def _split_into_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    pattern = r"(?<!\w\.\w)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"
    parts = re.split(pattern, text)
    return [p.strip() for p in parts if p.strip()]


def _run_llm_fact_check(claim: str, context: str) -> bool:
    """
    Return True if the LLM believes the claim is supported by the context.
    """
    system_prompt = (
        "You are a meticulous medical fact-checker. "
        "Given CONTEXT (evidence) and a CLAIM, decide whether the claim is "
        "fully supported by the context. Answer with only one word: 'Yes' or 'No'."
    )
    user_prompt = f"CONTEXT:\n{context}\n\nCLAIM:\n{claim}\n\nIs the claim fully supported?"

    payload: Dict[str, Any] = {
        "model": VLLM_SVA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 4,
        "temperature": 0.0,
    }
    headers = {"Content-Type": "application/json"}

    try:
        resp = requests.post(
            VLLM_SVA_ENDPOINT,
            json=payload,
            headers=headers,
            timeout=60,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip().lower()
        return text.startswith("yes")
    except Exception as e:
        print(f"WARNING: MedGemma fact-check call failed: {e}")
        # Fail-open: do not block the pipeline if the checker is down.
        return True


def run_safety_validation_agent(state: MaraGraphState) -> Dict[str, Any]:
    print("--- RUNNING SAFETY-VALIDATION AGENT (SVA) ---")

    report = (state.get("generated_report") or "").strip()
    retrieved_docs = state.get("retrieved_docs", [])

    if not report:
        error = {
            "error_type": "EmptyReport",
            "finding": "",
            "suggestion": "The Generation Agent did not produce a report.",
        }
        return {"validation_feedback": {"factuality_errors": [error]}}

    sentences = _split_into_sentences(report)
    if not sentences:
        sentences = [report]

    doc_texts = [d.get("content", "") for d in retrieved_docs if d.get("content")]
    if not doc_texts:
        print("SVA: No retrieved docs; skipping semantic check and passing.")
        return {"validation_feedback": None}

    doc_embeddings = _get_embeddings(doc_texts)
    sent_embeddings = _get_embeddings(sentences)

    cos_scores = util.cos_sim(sent_embeddings, doc_embeddings)
    max_scores, _ = torch.max(cos_scores, dim=1)

    potential_errors: List[str] = []
    for sentence, score in zip(sentences, max_scores):
        if float(score) < SIMILARITY_THRESHOLD:
            potential_errors.append(sentence)

    if not potential_errors:
        print("--- SVA PASSED (all sentences sufficiently supported) ---")
        return {"validation_feedback": None}

    # Use top-scoring docs from RA as context
    sorted_docs = sorted(
        retrieved_docs,
        key=lambda d: d.get("score", 0.0),
        reverse=True,
    )
    context_docs = [d.get("content", "") for d in sorted_docs[:3]]
    context_str = "\n\n".join(context_docs)

    final_errors: List[Dict[str, Any]] = []
    for claim in potential_errors:
        is_supported = _run_llm_fact_check(claim, context_str)
        if not is_supported:
            final_errors.append(
                {
                    "error_type": "FactualGroundingError",
                    "finding": claim,
                    "suggestion": (
                        "This sentence is not supported by the retrieved evidence. "
                        "Remove it or rephrase it to match the evidence."
                    ),
                }
            )

    if not final_errors:
        print("--- SVA PASSED after fact-checking ---")
        return {"validation_feedback": None}

    print(f"--- SVA FAILED: {len(final_errors)} factuality errors ---")
    return {"validation_feedback": {"factuality_errors": final_errors}}
