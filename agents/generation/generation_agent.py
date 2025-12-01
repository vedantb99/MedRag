# agents/generation/generation_agent.py
import requests
import json
from mara_pipelines.state import MaraGraphState # Import our contract

# --- vLLM Server Configuration ---
# This is the endpoint for the MedGemma model on vLLM
VLLM_GA_ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions" # Using OpenAI-compatible chat endpoint
MODEL_NAME = "google/medgemma-4b-it" # The actual model being served

def build_ga_prompt(state: MaraGraphState) -> tuple[str, str]:
    """
    Constructs the system and user prompts for the Generation Agent.
    Returns: (system_prompt, user_prompt)
    """
    system_prompt = (
        "You are an expert radiologist. Your task is to generate a clinical report "
        "based *only* on the provided context. Do not hallucinate or infer "
        "information not present in the context. Every finding must be "
        "traceable to the context. Be concise and accurate."
    )
    
    # 1. Collate retrieved context
    context_str = "\n".join(
        [f"CONTEXT {i+1}: {doc['content']}" for i, doc in enumerate(state['retrieved_docs'])]
    )
    
    # 2. Check if this is a refinement (C2FD loop)
    if state['refinement_count'] > 0:
        feedback = state['validation_feedback']
        history = state['refinement_history'][-1] # Get the last failed report
        
        user_prompt = f"""
        TASK: Regenerate a clinical report.
        PREVIOUS FAILED ATTEMPT:
        {history}
        
        SAFETY VALIDATION FEEDBACK (Errors you MUST fix):
        {json.dumps(feedback, indent=2)}
        
        Use the following retrieved context to generate a factually accurate report.
        You MUST ground your entire report in the provided context.
        
        CONTEXT:
        {context_str}
        
        USER QUERY: {state['clinical_query']}
        
        REFINED REPORT:
        """
    else:
        # This is the first pass (coarse generation)
        user_prompt = f"""
        TASK: Generate a clinical report based *only* on the provided context.
        
        CONTEXT:
        {context_str}
        
        USER QUERY: {state['clinical_query']}
        
        REPORT:
        """
    
    return system_prompt, user_prompt

def run_generation_agent(state: MaraGraphState) -> dict:
    """
    Runs the Generation Agent via vLLM.
    This function will replace 'mock_generation_agent'.
    """
    print("--- (REAL) RUNNING GENERATION AGENT (GA) ---")
    
    # 1. Build the prompt
    system_prompt, user_prompt = build_ga_prompt(state)
    
    # 2. Call the vLLM Server
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 1024,
        "temperature": 0.2, # Low temp for factual generation
    }
    
    try:
        response = requests.post(VLLM_GA_ENDPOINT, json=payload, headers=headers)
        response.raise_for_status() # Raise an exception for bad status codes
        
        generated_text = response.json()['choices'][0]['message']['content'].strip()
        
    except requests.exceptions.RequestException as e:
        print(f"ERROR: vLLM GA server call failed: {e}")
        # Fallback or error handling
        generated_text = "ERROR: Generation server is unavailable."

    print(f"GA Generated Report (Pass {state['refinement_count']}): {generated_text}")
    
    return {"generated_report": generated_text}