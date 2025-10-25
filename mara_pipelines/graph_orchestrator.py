# mara_pipelines/graph_orchestrator.py
# This is YOUR team's main file.
# You build the graph logic here.

from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Optional
import json
from .state import MaraGraphState

# --- 2. Import Agent Implementations (using MOCKS for now) ---
# As components are built, you'll swap these imports:
# from agents.retrieval.retrieval_agent import run_retrieval_agent
# from agents.generation.generation_agent import run_generation_agent
# ...etc.

from .mock_agents import (
    mock_retrieval_agent,
    mock_generation_agent,
    mock_safety_validation_agent,
    mock_refinement_agent
)

# --- 3. Define Graph Nodes ---
# Each node just calls its respective agent function
def retrieval_node(state: MaraGraphState):
    # This will call Team 1's code (RTX 3090)
    return mock_retrieval_agent(state)

def generation_node(state: MaraGraphState):
    # This will call your GA code (A100)
    return mock_generation_agent(state)

def safety_validation_node(state: MaraGraphState):
    # This will call your SVA code (RTX 3090)
    return mock_safety_validation_agent(state)

def refinement_node(state: MaraGraphState):
    # This will call your REFA code (A100)
    return mock_refinement_agent(state)

# --- 4. Define Graph Edges (The C2FD Loop Logic) ---
def should_continue(state: MaraGraphState):
    """
    This is the core of our Coarse-to-Fine Decoding (C2FD) loop.
    It checks the SVA's feedback.
    """
    max_refinements = 3 # Set a hard limit to prevent infinite loops
    
    if state['validation_feedback'] is None:
        # SVA passed, no feedback. We are done.
        return "end"
    elif state['refinement_count'] >= max_refinements:
        # We've tried too many times. Exit.
        print(f"--- WARNING: Max refinements ({max_refinements}) reached. Exiting loop. ---")
        return "end"
    else:
        # SVA found errors. Go to REFA.
        return "refine"

# --- 5. Build the Graph ---
def build_mara_graph():
    """Builds the LangGraph orchestrator."""
    
    workflow = StateGraph(MaraGraphState)
    
    # Add Nodes
    workflow.add_node("RA", retrieval_node)
    workflow.add_node("GA", generation_node)
    workflow.add_node("SVA", safety_validation_node)
    workflow.add_node("REFA", refinement_node)
    
    # Define the Flow
    workflow.set_entry_point("RA")
    
    # 1. After Retrieval -> Generate
    workflow.add_edge("RA", "GA")
    
    # 2. After Generation -> Validate
    workflow.add_edge("GA", "SVA")
    
    # 3. After Validation -> Decide (Conditional Edge)
    workflow.add_conditional_edges(
        "SVA",
        should_continue,
        {
            "end": END,         # If passed, end
            "refine": "REFA"    # If failed, go to refinement
        }
    )
    
    # 4. After Refinement -> Re-Generate
    # This creates the self-correction loop
    workflow.add_edge("REFA", "GA")
    
    # Compile the graph
    app = workflow.compile()
    return app

if __name__ == "__main__":
    print("Building MARA pipeline graph...")
    app = build_mara_graph()
    
    print("\n--- Running Pipeline (Pass 1) ---")
    
    # This is our initial RRG input
    initial_input = {
        "clinical_query": "65 y/o male with shortness of breath.",
        "image_path": "/images/cxr_001.png",
        "refinement_count": 0,
        "refinement_history": []
    }
    
    # Run the graph
    # The C2FD loop will run automatically
    final_state = app.invoke(initial_input)
    
    print("\n--- Pipeline Complete ---")
    print("\nFinal Generated Report:")
    print(final_state['generated_report'])
    
    print("\nRefinement History (Count):", final_state['refinement_count'])
    if final_state['refinement_count'] > 0:
        print("Previous (flawed) report:")
        print(json.dumps(final_state['refinement_history'], indent=2))