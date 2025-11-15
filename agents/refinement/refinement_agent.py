# agents/refinement/refinement_agent.py
from mara_pipelines.state import MaraGraphState
import json

def run_refinement_agent(state: MaraGraphState) -> dict:
    """
    Runs the Refinement Agent (REFA).
    This agent preps the state for the next C2FD loop.
    """
    print("--- (REAL) RUNNING REFINEMENT AGENT (REFA) ---")
    
    # 1. Log the failed report
    history = state.get('refinement_history', [])
    history.append(state['generated_report'])
    
    # 2. Increment the loop counter
    count = state.get('refinement_count', 0) + 1
    
    print(f"REFA: Failed attempt {count}. Feedback received:")
    print(json.dumps(state['validation_feedback'], indent=2))
    
    # The 'clinical_query' is *not* overwritten here.
    # The build_ga_prompt() function is responsible for seeing
    # the feedback and constructing a new prompt.
    
    return {
        "refinement_history": history,
        "refinement_count": count
        # Note: We pass 'validation_feedback' through for the GA to use
    }