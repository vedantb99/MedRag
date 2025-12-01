from typing import Any, Dict, List, Optional, TypedDict


class MaraGraphState(TypedDict, total=False):
    """
    Shared state passed between all MedRAG agents.
    LangGraph nodes read from and write to this structure.
    """

    # Input
    clinical_query: str            # Free-text clinical info / indication
    image_path: str                # Path to the CXR image (for Vision Agent)

    # Vision Agent (VA)
    image_embeddings: Optional[Any]  # BiomedCLIP image embeddings (torch.Tensor)
    image_features: Optional[Dict[str, Any]]  # Metadata about image features
    image_text_scores: Optional[Any]  # Image-text similarity scores

    # Retrieval Agent (RA)
    retrieved_docs: List[Dict[str, Any]]  # [{"content": str, "score": float, ...}, ...]

    # Generation Agent (GA)
    generated_report: str          # Draft / refined report text

    # Safety-Validation Agent (SVA)
    # If None, validation passed; otherwise structured error info.
    validation_feedback: Optional[Dict[str, Any]]

    # Refinement Agent (REFA)
    refinement_history: List[str]  # Previous generated reports
    refinement_count: int          # Number of refinement iterations so far
