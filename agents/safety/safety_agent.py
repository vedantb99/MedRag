# agents/safety/safety_agent.py
from mara_pipelines.state import MaraGraphState
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import util
import torch
import requests
import json
import re # For sentence splitting

# --- vLLM Server Configuration ---
VLLM_SVA_ENDPOINT = "http://127.0.0.1:8001/v1/completions" # A *different* port for the SVA model

# --- SVA Component 1: Fast Pass (ClinicalBERT) ---
# This part is unchanged and runs on the same machine (A100)
SVA_TOKENIZER = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
SVA_MODEL = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
SVA_MODEL.to("cuda:0") # We have A100s, let's use them!

def get_embeddings(texts, tokenizer, model):
    """Helper function to get sentence embeddings."""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1) # Mean pooling

def run_llm_fact_check(claim, context_str, model_name="google/medgemma"):
    """
    Performs the "Deep Pass" LLM fact-check against a vLLM server.
    """
    system_prompt = (
        "You are a meticulous fact-checker. Evaluate whether the 'CLAIM' "
        "is *fully and explicitly* supported by the 'CONTEXT'. "
        "Respond with only 'Yes' or 'No'."
    )
    user_prompt = f"CONTEXT: {context_str}\n\nCLAIM: {claim}"
    
    # Using MedGemma template (example)
    prompt = f"<start_of_turn>user\n{system_prompt}\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"

    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": 5, # Just needs to say "Yes" or "No"
        "temperature": 0.0,
    }
    
    try:
        response = requests.post(VLLM_SVA_ENDPOINT, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()['choices'][0]['text'].strip().lower()
        return "yes" in result
    except requests.exceptions.RequestException as e:
        print(f"ERROR: vLLM SVA server call failed: {e}")
        return False # Fail-safe: if checker fails, assume the fact is unverified

def run_safety_validation_agent(state: MaraGraphState) -> dict:
    """
    Runs the hybrid Safety Validation Agent (SVA) on the A100.
    """
    print("--- (REAL) RUNNING SAFETY VALIDATION AGENT (SVA) ---")
    
    report = state['generated_report']
    retrieved_docs = [doc['content'] for doc in state['retrieved_docs']]
    context_str = "\n".join(retrieved_docs)
    
    # Split report into individual sentences (findings)
    # A simple regex split is better than .split('. ')
    report_sentences = [s.strip() for s in re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', report) if s.strip()]
    
    if not report_sentences:
        return {"validation_feedback": {"factuality_errors": [{"error_type": "Empty Report", "suggestion": "Generator produced an empty report."}]}}

    # --- 1. Fast Pass: Semantic Similarity Check ---
    doc_embeddings = get_embeddings(retrieved_docs, SVA_TOKENIZER, SVA_MODEL)
    report_embeddings = get_embeddings(report_sentences, SVA_TOKENIZER, SVA_MODEL)
    
    similarity_threshold = 0.7 # Tunable
    potential_errors = []
    
    cos_scores = util.cos_sim(report_embeddings, doc_embeddings)
    max_scores = torch.max(cos_scores, dim=1).values

    for i, max_score in enumerate(max_scores):
        if max_score < similarity_threshold:
            potential_errors.append(report_sentences[i])

    # --- 2. Deep Pass: LLM Fact-Check ---
    # We only run the expensive LLM check on sentences that
    # failed the fast, cheap semantic check.
    final_errors = []
    
    if potential_errors:
        print(f"SVA: Fast pass failed for {len(potential_errors)} sentences. Starting deep check...")
        for claim in potential_errors:
            is_supported = run_llm_fact_check(claim, context_str)
            if not is_supported:
                final_errors.append({
                    "error_type": "Factual Grounding Error (LLM Confirmed)",
                    "finding": claim,
                    "suggestion": "This finding is not supported by the retrieved context. Remove it or re-generate."
                })
    
    # --- 3. Formulate Feedback ---
    if not final_errors:
        print("--- SVA PASSED ---")
        return {"validation_feedback": None} # PASS!
    else:
        print(f"--- SVA FAILED: {len(final_errors)} errors found. ---")
        return {"validation_feedback": {"factuality_errors": final_errors}}