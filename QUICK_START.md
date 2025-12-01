# MedRAG Quick Start Guide

## Prerequisites
```bash
# Make sure vLLM server is running
vllm serve "google/medgemma-4b-it" --gpu-memory-utilization 0.9
```

## Basic Usage

### 1. Text-Only Query (No Image)
```bash
python main.py --clinical_query "65-year-old male with shortness of breath and fever"
```

### 2. With Chest X-ray Image
```bash
python main.py \
  --clinical_query "Patient with chest pain and dyspnea" \
  --image_path "data/test_xray.png"
```

### 3. Complex Clinical Query
```bash
python main.py --clinical_query \
  "78-year-old female with chronic cough, weight loss, night sweats, and history of tuberculosis"
```

## What Happens Behind the Scenes

### Pipeline Flow:
```
User Query + Image (optional)
        ↓
[Vision Agent] - Extracts image embeddings with BiomedCLIP
        ↓
[Retrieval Agent] - Searches 30 medical documents with Bio-E5
        ↓
[Generation Agent] - Creates report using MedGemma-4B
        ↓
[Safety Validation] - Checks factual correctness
        ↓
[Refinement if needed] - C2FD loop for corrections
        ↓
Final Report
```

## Components

### Vision Agent (NEW! ✨)
- **Model**: BiomedCLIP (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
- **Input**: Chest X-ray images (PNG, JPG)
- **Output**: 512-dimensional embeddings
- **Device**: CUDA
- **Note**: Gracefully skips if no image provided

### Retrieval Agent (UPGRADED! 📈)
- **Model**: Bio-E5 (intfloat/e5-base-v2)
- **Corpus**: 30 medical documents in `data/knowledge_base.jsonl`
  - 20 radiology findings
  - 10 clinical knowledge entries
- **Device**: CPU (to save GPU memory)
- **Top-K**: 8 documents

### Generation Agent
- **Model**: MedGemma-4B (via vLLM)
- **Endpoint**: http://127.0.0.1:8000/v1/chat/completions
- **Temperature**: 0.2 (factual generation)

### Safety Validation Agent
- **Model**: Bio_ClinicalBERT (emilyalsentzer/Bio_ClinicalBERT)
- **Threshold**: 0.7 similarity
- **Device**: CPU

## Example Queries to Try

### 1. Pneumonia
```bash
python main.py --clinical_query \
  "45-year-old with fever, productive cough, and right-sided chest pain"
```

### 2. Pneumothorax
```bash
python main.py --clinical_query \
  "Patient with sudden onset chest pain and shortness of breath"
```

### 3. Heart Failure
```bash
python main.py --clinical_query \
  "Elderly patient with dyspnea, orthopnea, and bilateral leg swelling"
```

### 4. Tuberculosis
```bash
python main.py --clinical_query \
  "Chronic cough, weight loss, night sweats, and fever"
```

### 5. Lung Cancer
```bash
python main.py --clinical_query \
  "60-year-old smoker with hemoptysis and weight loss"
```

## Testing Refinement Loop

To trigger the refinement loop, use stricter validation:

```bash
export MEDRAG_SVA_SIM_THRESHOLD=0.85
python main.py --clinical_query "Patient with vague symptoms"
```

## Configuration Options

### Retrieval Settings
```bash
export MEDRAG_RETRIEVAL_MODEL="intfloat/e5-base-v2"
export MEDRAG_CORPUS_JSONL="data/pubmed.jsonl"  # 2,584 PubMed abstracts
export MEDRAG_RETRIEVAL_TOP_K=8
export MEDRAG_RETRIEVAL_DEVICE="cpu"

# Re-ranking (improves accuracy)
export MEDRAG_USE_COLBERT_RERANK=true  # Enable ColBERT-style re-ranking
export MEDRAG_RERANK_TOP_K=20          # Candidates before re-ranking
```

### Safety Validation
```bash
export MEDRAG_SVA_SIM_THRESHOLD=0.7  # Lower = more lenient
export MEDRAG_SVA_DEVICE="cpu"
```

### Refinement
```bash
export MEDRAG_MAX_REFINEMENTS=3
```

## Understanding the Output

```
================ MEDRAG FINAL REPORT ================

[Generated radiology report here]

=====================================================

--- Debug information ---
Refinement count: 0              # Number of C2FD iterations
# Retrieved docs: 8              # Documents used for generation
```

### Refinement Count:
- `0`: Report passed validation on first try ✅
- `1+`: Report needed refinement (C2FD loop activated)
- `3+`: Hit max refinements limit

## Troubleshooting

### GPU Out of Memory
**Solution**: Components are optimized for 24GB GPUs
- vLLM + BiomedCLIP on GPU (~10-12 GB)
- Bio-E5 + Bio_ClinicalBERT on CPU

### vLLM Server Not Running
```
ERROR: vLLM GA server call failed: 404
```
**Solution**: Start vLLM server first in separate terminal

### No Image Provided
```
VA: No image path provided, skipping vision processing.
```
**Note**: This is normal! Vision agent is optional.

### Retrieval Returns Few Documents
```
# Retrieved docs: 2
```
**Solution**: Add more documents to `data/knowledge_base.jsonl`

## Advanced Usage

### Using Real Chest X-rays
```bash
# Download from NIH dataset or use MIMIC-CXR
python main.py \
  --clinical_query "Evaluate for infiltrates" \
  --image_path "/path/to/real_cxr.dcm"
```

### Custom Corpus
```bash
# Create your own knowledge base
export MEDRAG_CORPUS_JSONL="/path/to/custom_corpus.jsonl"
python main.py --clinical_query "..."
```

### Adjust Retrieval Weights
```python
# In retrieval_agent.py
_DENSE_WEIGHT = 0.7  # Dense embedding weight
_LEXICAL_WEIGHT = 0.3  # Lexical overlap weight
```

## Performance Tips

1. **First run is slow**: Models need to download (~1.5 GB total)
2. **Subsequent runs are fast**: Models are cached
3. **Vision agent adds ~1-2s**: Image processing overhead
4. **CPU inference is slower**: But necessary to fit in GPU memory

## Resources

- **Full Documentation**: See `ENHANCEMENT_SUMMARY.md`
- **Vision Testing**: See `data/VISION_AGENT_TESTING.md`
- **Project Report**: See `genai_phase1 (1) (2).pdf`

## Quick Validation

Test everything is working:
```bash
# 1. Text-only
python main.py --clinical_query "Patient with fever"

# 2. With image
python main.py \
  --clinical_query "Patient with fever" \
  --image_path "data/test_xray.png"

# Both should complete successfully!
```

---

**Status**: All features working! ✅
**Last Updated**: November 29, 2025
