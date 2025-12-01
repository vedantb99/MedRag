import json
from typing import Any, Dict

from mara_pipelines.state import MaraGraphState


def run_refinement_agent(state: MaraGraphState) -> Dict[str, Any]:
    """
    Update refinement history and counter so GA can perform a corrected pass.
    """
    print("--- RUNNING REFINEMENT AGENT (REFA) ---")

    history = list(state.get("refinement_history", []))
    history.append(state.get("generated_report", ""))

    count = int(state.get("refinement_count", 0)) + 1

    print(f"REFA: Failed attempt {count}. Feedback received:")
    print(json.dumps(state.get("validation_feedback"), indent=2))

    return {
        "refinement_history": history,
        "refinement_count": count,
    }
