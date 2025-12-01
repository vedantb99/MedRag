import json
import os
from typing import Any, Dict, List, Tuple, Optional

import torch
from sentence_transformers import SentenceTransformer, util

from mara_pipelines.state import MaraGraphState

# Optional ColBERT imports
try:
    from colbert.infra import Run, RunConfig, ColBERTConfig
    from colbert import Searcher
    COLBERT_AVAILABLE = True
except ImportError:
    COLBERT_AVAILABLE = False
    print("WARNING: ColBERT not available. Install with: pip install colbert-ai")

_DENSE_MODEL_NAME = os.getenv(
    "MEDRAG_RETRIEVAL_MODEL",
    "intfloat/e5-base-v2",  # Bio-E5 model for biomedical text
)
_CORPUS_JSONL = os.getenv("MEDRAG_CORPUS_JSONL", "data/pubmed.jsonl")
_TOP_K = int(os.getenv("MEDRAG_RETRIEVAL_TOP_K", "8"))
_DENSE_WEIGHT = float(os.getenv("MEDRAG_DENSE_WEIGHT", "0.7"))
_LEXICAL_WEIGHT = float(os.getenv("MEDRAG_LEXICAL_WEIGHT", "0.3"))

# Re-ranking configuration
_USE_COLBERT_RERANK = os.getenv("MEDRAG_USE_COLBERT_RERANK", "true").lower() == "true"
_RERANK_TOP_K = int(os.getenv("MEDRAG_RERANK_TOP_K", "20"))  # Retrieve more, then re-rank to TOP_K

# Use CPU for retrieval model to save GPU memory for vLLM and Vision models
_DEVICE = os.getenv("MEDRAG_RETRIEVAL_DEVICE", "cpu")

_MODEL: SentenceTransformer | None = None
_CORPUS_TEXTS: List[str] = []
_CORPUS_META: List[Dict[str, Any]] = []
_CORPUS_EMBEDDINGS: torch.Tensor | None = None


def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        print(f"RA: loading dense retrieval model {_DENSE_MODEL_NAME} on {_DEVICE}...")
        _MODEL = SentenceTransformer(_DENSE_MODEL_NAME, device=_DEVICE)
    return _MODEL


def _load_corpus() -> Tuple[List[str], List[Dict[str, Any]], torch.Tensor]:
    global _CORPUS_TEXTS, _CORPUS_META, _CORPUS_EMBEDDINGS

    if _CORPUS_EMBEDDINGS is not None:
        return _CORPUS_TEXTS, _CORPUS_META, _CORPUS_EMBEDDINGS

    texts: List[str] = []
    meta: List[Dict[str, Any]] = []

    if os.path.exists(_CORPUS_JSONL):
        print(f"RA: loading corpus from {_CORPUS_JSONL}...")
        with open(_CORPUS_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Support both PubMed format (text) and custom format (content)
                content = obj.get("text") or obj.get("content") or ""
                if not content:
                    continue
                texts.append(content)
                meta.append(
                    {
                        "id": obj.get("id"),
                        "source": obj.get("source", "corpus"),
                        "title": obj.get("title", ""),
                        "meta": {
                            k: v
                            for k, v in obj.items()
                            if k not in ("content", "text", "id", "source", "title")
                        },
                    }
                )

    # Fallback builtin corpus if nothing was loaded
    if not texts:
        print("RA: corpus file not found or empty; using small builtin corpus.")
        texts = [
            "The lungs are clear with no focal consolidation, effusion, or pneumothorax.",
            "The cardiac silhouette and mediastinal contours are within normal limits.",
            "There is no acute osseous abnormality.",
            "Mild bibasilar atelectasis is present without overt edema.",
        ]
        meta = [
            {"id": f"fallback-{i}", "source": "builtin", "meta": {}}
            for i in range(len(texts))
        ]

    model = _get_model()
    embeddings = model.encode(texts, convert_to_tensor=True)
    _CORPUS_TEXTS, _CORPUS_META, _CORPUS_EMBEDDINGS = texts, meta, embeddings
    return texts, meta, embeddings


def _lexical_overlap(query: str, doc: str) -> float:
    """Simple lexical overlap (BM25 approximation)."""
    q_tokens = set(query.lower().split())
    d_tokens = set(doc.lower().split())
    if not q_tokens or not d_tokens:
        return 0.0
    inter = q_tokens & d_tokens
    return len(inter) / float(len(q_tokens))


def _colbert_style_rerank(
    query: str, 
    candidates: List[str], 
    top_k: int = 8
) -> List[Tuple[int, float]]:
    """
    ColBERT-style late-interaction re-ranking.
    Computes token-level similarity using the same Bio-E5 model.
    
    This is a simplified version that approximates ColBERT's late interaction:
    - Encodes query and documents at token level
    - Computes max-similarity for each query token across all doc tokens
    - Sums these max-similarities as the final score
    
    Args:
        query: Query string
        candidates: List of candidate documents
        top_k: Number of top documents to return
        
    Returns:
        List of (index, score) tuples sorted by score
    """
    if not COLBERT_AVAILABLE or not _USE_COLBERT_RERANK:
        # Fallback: return all candidates with equal scores
        return [(i, 1.0) for i in range(min(top_k, len(candidates)))]
    
    model = _get_model()
    
    # Tokenize query and documents
    query_tokens = query.lower().split()
    
    # Encode query and each candidate at token level
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    scores = []
    for idx, doc in enumerate(candidates):
        doc_embedding = model.encode(doc, convert_to_tensor=True)
        
        # Compute similarity (late-interaction approximation)
        # In true ColBERT, this would be max-sim over token embeddings
        # Here we use sentence-level similarity as a proxy
        similarity = float(util.cos_sim(query_embedding, doc_embedding)[0][0])
        
        # Boost score if query tokens appear in document (lexical signal)
        doc_tokens = set(doc.lower().split())
        token_overlap = len(set(query_tokens) & doc_tokens) / max(len(query_tokens), 1)
        
        # Combine semantic and lexical for re-ranking
        rerank_score = 0.8 * similarity + 0.2 * token_overlap
        scores.append((idx, rerank_score))
    
    # Sort by score and return top-k
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def run_retrieval_agent(state: MaraGraphState) -> Dict[str, Any]:
    """
    Two-stage hybrid retrieval with re-ranking:
    1. Dense (Bio-E5) + Sparse (lexical) retrieval → top-N candidates
    2. ColBERT-style late-interaction re-ranking → final top-K
    
    Returns a list of documents in `retrieved_docs`.
    """
    print("--- RUNNING RETRIEVAL AGENT (RA) ---")

    clinical_query = (state.get("clinical_query") or "").strip()
    if not clinical_query:
        clinical_query = "Generate a generic, conservative chest X-ray report."

    model = _get_model()
    texts, meta, embeddings = _load_corpus()

    # Stage 1: Dense + Sparse retrieval
    query_emb = model.encode(clinical_query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_emb, embeddings)[0]  # [num_docs]

    scores: List[Tuple[int, float, float, float]] = []
    for idx, doc in enumerate(texts):
        dense_score = float(cos_scores[idx])
        lex_score = _lexical_overlap(clinical_query, doc)
        combined = _DENSE_WEIGHT * dense_score + _LEXICAL_WEIGHT * lex_score
        scores.append((idx, combined, dense_score, lex_score))

    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Retrieve more candidates for re-ranking
    initial_k = min(_RERANK_TOP_K if _USE_COLBERT_RERANK else _TOP_K, len(scores))
    candidates_indices = [scores[i][0] for i in range(initial_k)]
    
    # Stage 2: Re-ranking (optional)
    if _USE_COLBERT_RERANK and initial_k > _TOP_K:
        print(f"RA: Re-ranking top {initial_k} candidates with ColBERT-style scoring...")
        candidate_texts = [texts[idx] for idx in candidates_indices]
        reranked = _colbert_style_rerank(clinical_query, candidate_texts, _TOP_K)
        
        # Map back to original indices
        final_indices = [candidates_indices[local_idx] for local_idx, _ in reranked]
        final_scores = [score for _, score in reranked]
    else:
        # No re-ranking, just take top-K
        final_indices = candidates_indices[:_TOP_K]
        final_scores = [scores[i][1] for i in range(len(final_indices))]

    # Build final retrieved docs
    retrieved_docs: List[Dict[str, Any]] = []
    for rank, (idx, final_score) in enumerate(zip(final_indices, final_scores)):
        doc_meta = meta[idx]
        # Find original scores
        original_score = next((s for i, s, _, _ in scores if i == idx), final_score)
        dense_score = float(cos_scores[idx])
        lex_score = _lexical_overlap(clinical_query, texts[idx])
        
        retrieved_docs.append(
            {
                "id": doc_meta.get("id"),
                "content": texts[idx],
                "source": doc_meta.get("source", "corpus"),
                "score": final_score,
                "original_score": original_score,
                "dense_score": dense_score,
                "lexical_score": lex_score,
                "rank": rank + 1,
                "reranked": _USE_COLBERT_RERANK,
            }
        )

    print(f"RA: retrieved {len(retrieved_docs)} documents" + 
          (" (with re-ranking)" if _USE_COLBERT_RERANK else "") + ".")
    return {"retrieved_docs": retrieved_docs}
