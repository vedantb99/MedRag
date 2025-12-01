# mara_pipelines/mock_agents.py
# These are your black-box stand-ins.
# Team 1 (RA) and your own team (GA/SVA) will replace these
# with real component calls.

import time
from .state import MaraGraphState
def mock_retrieval_agent(state: MaraGraphState) -> MaraGraphState:
    """
    MOCK Black Box for Team 1 (RAG/RA).
    This will eventually be a complex call to their ColBERTv2 re-ranker
    running on the RTX 3090.
    """
    print("--- (MOCK) RUNNING RETRIEVAL AGENT (RA) ---")
    time.sleep(0.5) # Simulate work
    docs = [
        {"content": "Finding: Mild pulmonary edema is noted.", "score": 0.95, "source": "doc_1"},
        {"content": "Finding: The cardiomediastinal silhouette is enlarged.", "score": 0.92, "source": "doc_2"},
        {"content": "Finding: No evidence of pneumothorax.", "score": 0.88, "source": "doc_3"},
    ]
    return {"retrieved_docs": docs}

def mock_generation_agent(state: MaraGraphState) -> MaraGraphState:
    """
    MOCK Black Box for your own GA.
    This will eventually be your Vertex AI RAG Engine (Gemini Pro) call,
    running on the A100.
    """
    print("--- (MOCK) RUNNING GENERATION AGENT (GA) ---")
    time.sleep(1.0) # Simulate work
    
    # Simulate first pass vs. refinement
    if state['refinement_count'] == 0:
        report = "FINDINGS: Cardiomegaly is present. Mild pulmonary edema. No pneumothorax. IMPRESSION: Signs of congestive heart failure."
    else:
        # Simulate a refined report based on SVA feedback
        report = "FINDINGS: The cardiomediastinal silhouette is enlarged, consistent with cardiomegaly. Mild interstitial pulmonary edema is noted. No pneumothorax. IMPRESSION: Findings suggest congestive heart failure."
    
    return {"generated_report": report}

def mock_safety_validation_agent(state: MaraGraphState) -> MaraGraphState:
    """
    MOCK Black Box for your own SVA.
    This will eventually be your hybrid ClinicalBERT + LLM fact-checker
    running on the RTX 3090.
    """
    print("--- (MOCK) RUNNING SAFETY VALIDATION AGENT (SVA) ---")
    time.sleep(0.7) # Simulate work
    
    # Simulate SVA finding an error on the first pass
    if state['refinement_count'] == 0:
        feedback = {
            "factuality_errors": [
                {"finding": "Cardiomegaly is present.", "ground_truth": "The cardiomediastinal silhouette is enlarged."}
            ],
            "suggestions": ["Refine 'Cardiomegaly is present' to be more descriptive, e.g., 'The cardiomediastinal silhouette is enlarged.'"]
        }
        return {"validation_feedback": feedback}
    else:
        # On the second pass, the report is good.
        print("--- SVA PASSED ---")
        return {"validation_feedback": None} # None == PASSED

def mock_refinement_agent(state: MaraGraphState) -> MaraGraphState:
    """
    MOCK Black Box for your REFA.
    This node's job is to prep the state for the *next* GA call.
    It takes SVA feedback and formats a new prompt for the GA.
    """
    print("--- (MOCK) RUNNING REFINEMENT AGENT (REFA) ---")
    
    # Log the bad report
    history = state.get('refinement_history', [])
    history.append(state['generated_report'])
    
    # Create a new prompt for the GA based on SVA feedback
    # (In the real agent, you'll build a detailed prompt here)
    new_query = f"Original Query: {state['clinical_query']}. \
                  Previous attempt was flawed. \
                  Feedback: {state['validation_feedback']}. \
                  Regenerate the report, incorporating this feedback."

    return {
        "refinement_history": history,
        "refinement_count": state['refinement_count'] + 1,
        "clinical_query": new_query # Overwrite the query for the GA
    }