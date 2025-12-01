# MedRAG: Multi-Agent Retrieval-Augmented Radiology Report Generation

This project implements **MedRAG**, a multi-agent system for clinically grounded
radiology report generation.

Architecture (high level):

- **Retrieval Agent (RA)**  
  Hybrid dense + lexical retrieval over a biomedical corpus (JSONL file) using
  SentenceTransformers.

- **Generation Agent (GA)**  
  Calls a large language model (e.g., Qwen2-72B-Instruct) served by **vLLM**
  via an OpenAI-compatible `/v1/completions` endpoint.

- **Safety–Validation Agent (SVA)**  
  2-stage factuality checking:
  1. Semantic similarity using Bio_ClinicalBERT
  2. LLM-based fact checking (e.g., MedGemma via vLLM)

- **Refinement Agent (REFA)**  
  Implements a coarse-to-fine decoding loop (C2FD). If SVA flags issues, GA is
  prompted again with structured feedback until the report passes or a maximum
  number of iterations is reached.

Orchestration is done with **LangGraph** and a shared `MaraGraphState`.

---

## Project Structure

```text
MedRag-main/
  main.py                      # CLI entry point
  requirements.txt
  README.md

  mara_pipelines/
    __init__.py
    state.py                   # Typed central state (MaraGraphState)
    graph_orchestrator.py      # LangGraph workflow (RA → GA → SVA → REFA)

  agents/
    __init__.py
    retrieval/
      __init__.py
      retrieval_agent.py       # Hybrid retrieval
    generation/
      __init__.py
      generation_agent.py      # vLLM LLM caller
    safety/
      __init__.py
      safety_agent.py          # ClinicalBERT + MedGemma fact checking
    refinement/
      __init__.py
      refinement_agent.py      # Maintains refinement loop state

  configs/
    base_config.yaml
    hardware_3090.yaml
    hardware_a100.yaml

  data_processing/
    process_mimic.py           # Simple CSV → JSONL corpus conversion (optional)
    build_colbert_index.py     # Placeholder / optional indexing script
