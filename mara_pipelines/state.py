# mara_pipelines/state.py
from typing import List, Dict, TypedDict, Optional

class MaraGraphState(TypedDict):
    """
    Defines the central state for the MARA pipeline.
    This is the "contract" all agents must adhere to.
    """
    # --- Input ---
    clinical_query: str                 # The initial clinical prompt/notes
    image_path: str                     # Path to the CXR image for multimodal RRG
    
    # --- Retrieval Agent (RA) State ---
    retrieved_docs: List[Dict[str, any]] # The list of retrieved documents from ColBERTv2
                                        # Each dict: {"content": "...", "score": 0.9, "source": "..."}
    
    # --- Generation Agent (GA) State ---
    generated_report: str               # The draft report from Gemini Pro
    
    # --- Safety/Validation (SVA) State ---
    validation_feedback: Optional[Dict[str, any]] # SVA's findings.
                                                  # If None, it means validation passed.
    
    # --- Refinement (REFA) State ---
    refinement_history: List[str]       # A log of previous generation attempts
    refinement_count: int               # Counter for the C2FD loop