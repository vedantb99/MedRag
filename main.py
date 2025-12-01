import argparse
from mara_pipelines.graph_orchestrator import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MedRAG: multi-agent RAG pipeline for radiology report generation."
    )
    parser.add_argument(
        "-q",
        "--clinical_query",
        required=True,
        type=str,
        help="Clinical query / indication / notes.",
    )
    parser.add_argument(
        "-i",
        "--image_path",
        default="",
        type=str,
        help="Optional path to chest X-ray image (reserved for Vision Agent).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    final_state = run_pipeline(
        clinical_query=args.clinical_query,
        image_path=args.image_path,
    )

    print("\n================ MEDRAG FINAL REPORT ================\n")
    print(final_state.get("generated_report", "").strip())
    print("\n=====================================================")

    print("\n--- Debug information ---")
    print(f"Refinement count: {final_state.get('refinement_count', 0)}")
    print(f"# Retrieved docs: {len(final_state.get('retrieved_docs', []))}")
    if final_state.get("validation_feedback") is not None:
        import json as _json

        print("\nValidation feedback (final iteration):")
        print(_json.dumps(final_state["validation_feedback"], indent=2))


if __name__ == "__main__":
    main()


# Example usage:
# python main.py \
#   --clinical_query "65-year-old male with shortness of breath and fever" \
#   --image_path "/path/to/cxr.png"   # optional, not yet used by code