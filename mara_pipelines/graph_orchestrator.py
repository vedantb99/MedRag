import os
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from mara_pipelines.state import MaraGraphState
from agents.retrieval.retrieval_agent import run_retrieval_agent
from agents.generation.generation_agent import run_generation_agent
from agents.safety.safety_agent import run_safety_validation_agent
from agents.refinement.refinement_agent import run_refinement_agent
from agents.vision.vision_agent import run_vision_agent


def vision_node(state: MaraGraphState) -> Dict[str, Any]:
    """VA: process chest X-ray image and extract embeddings."""
    return run_vision_agent(state)


def retrieval_node(state: MaraGraphState) -> Dict[str, Any]:
    """RA: populate `retrieved_docs`."""
    return run_retrieval_agent(state)


def generation_node(state: MaraGraphState) -> Dict[str, Any]:
    """GA: populate / update `generated_report`."""
    return run_generation_agent(state)


def safety_validation_node(state: MaraGraphState) -> Dict[str, Any]:
    """SVA: either return `validation_feedback=None` or structured errors."""
    return run_safety_validation_agent(state)


def refinement_node(state: MaraGraphState) -> Dict[str, Any]:
    """REFA: log failed report and increment refinement counter."""
    return run_refinement_agent(state)


def should_continue(state: MaraGraphState) -> str:
    """
    Routing function for LangGraph.

    If `validation_feedback` is None → end the graph.
    Otherwise, loop through REFA → GA again, up to max_refinements.
    """
    max_refinements = int(os.getenv("MEDRAG_MAX_REFINEMENTS", "3"))
    feedback = state.get("validation_feedback")

    # No feedback => validation passed
    if not feedback:
        return "end"

    count = int(state.get("refinement_count", 0))
    if count >= max_refinements:
        print(
            f"--- WARNING: Max refinements ({max_refinements}) reached. "
            "Terminating C2FD loop."
        )
        return "end"

    return "refine"


def build_app():
    """Build and compile the LangGraph workflow."""
    workflow = StateGraph(MaraGraphState)

    # Add vision agent node
    workflow.add_node("VA", vision_node)
    
    workflow.add_node("RA", retrieval_node)
    workflow.add_node("GA", generation_node)
    workflow.add_node("SVA", safety_validation_node)
    workflow.add_node("REFA", refinement_node)

    # Start with Vision Agent (it will skip if no image)
    workflow.set_entry_point("VA")
    workflow.add_edge("VA", "RA")
    workflow.add_edge("RA", "GA")
    workflow.add_edge("GA", "SVA")

    workflow.add_conditional_edges(
        "SVA",
        should_continue,
        {
            "end": END,
            "refine": "REFA",
        },
    )

    workflow.add_edge("REFA", "GA")

    return workflow.compile()


def run_pipeline(clinical_query: str, image_path: str = "") -> MaraGraphState:
    """
    Convenience entry point for the whole MedRAG pipeline.

    Returns the final `MaraGraphState` after running the graph.
    """
    app = build_app()
    initial_state: MaraGraphState = {
        "clinical_query": clinical_query,
        "image_path": image_path,
        "retrieved_docs": [],
        "generated_report": "",
        "validation_feedback": None,
        "refinement_history": [],
        "refinement_count": 0,
    }
    final_state = app.invoke(initial_state)
    return final_state
