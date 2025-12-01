"""
Simple helper to convert a MIMIC-style CSV into a JSONL corpus
for MedRAG retrieval (id, source, content).
"""

import argparse
import csv
import json
from pathlib import Path


def process_mimic_csv(
    input_csv: str,
    output_jsonl: str,
    id_column: str = "study_id",
    text_column: str = "report",
    source: str = "mimic-cxr",
) -> None:
    in_path = Path(input_csv)
    out_path = Path(output_jsonl)

    with in_path.open("r", encoding="utf-8") as fin, out_path.open(
        "w", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin)
        for row in reader:
            text = (row.get(text_column) or "").strip()
            if not text:
                continue
            obj = {
                "id": row.get(id_column),
                "source": source,
                "content": text,
            }
            fout.write(json.dumps(obj) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MIMIC-style CSV into MedRAG JSONL corpus."
    )
    parser.add_argument("-i", "--input_csv", required=True)
    parser.add_argument("-o", "--output_jsonl", required=True)
    parser.add_argument("--id_column", default="study_id")
    parser.add_argument("--text_column", default="report")
    parser.add_argument("--source", default="mimic-cxr")
    args = parser.parse_args()
    process_mimic_csv(
        args.input_csv,
        args.output_jsonl,
        id_column=args.id_column,
        text_column=args.text_column,
        source=args.source,
    )


if __name__ == "__main__":
    main()
