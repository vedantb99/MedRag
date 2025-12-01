# agents/vision/vision_agent.py
"""
Vision Agent for MedRAG
Handles chest X-ray image processing using BiomedCLIP
Extracts image embeddings for multimodal retrieval and generation
"""
import os
from typing import Any, Dict, Optional

import torch
import open_clip
from PIL import Image

from mara_pipelines.state import MaraGraphState

# Model configuration
_BIOMEDCLIP_MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global model cache
_VISION_MODEL = None
_VISION_PREPROCESS = None
_TOKENIZER = None


def _load_biomedclip():
    """
    Load BiomedCLIP model for medical image understanding.
    Returns: (model, preprocess_fn, tokenizer)
    """
    global _VISION_MODEL, _VISION_PREPROCESS, _TOKENIZER
    
    if _VISION_MODEL is None:
        print(f"Vision Agent: Loading BiomedCLIP model on {_DEVICE}...")
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                _BIOMEDCLIP_MODEL_NAME
            )
            tokenizer = open_clip.get_tokenizer(_BIOMEDCLIP_MODEL_NAME)
            
            model = model.to(_DEVICE)
            model.eval()
            
            _VISION_MODEL = model
            _VISION_PREPROCESS = preprocess
            _TOKENIZER = tokenizer
            
            print("Vision Agent: BiomedCLIP loaded successfully")
        except Exception as e:
            print(f"WARNING: Failed to load BiomedCLIP: {e}")
            print("Vision Agent will be disabled for this run.")
            return None, None, None
    
    return _VISION_MODEL, _VISION_PREPROCESS, _TOKENIZER


def extract_image_embeddings(image_path: str) -> Optional[torch.Tensor]:
    """
    Extract image embeddings from a chest X-ray using BiomedCLIP.
    
    Args:
        image_path: Path to the chest X-ray image
        
    Returns:
        Image embedding tensor or None if processing fails
    """
    model, preprocess, _ = _load_biomedclip()
    
    if model is None or not os.path.exists(image_path):
        return None
    
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(_DEVICE)
        
        # Extract embeddings
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu()
    
    except Exception as e:
        print(f"ERROR: Failed to process image {image_path}: {e}")
        return None


def get_image_text_similarity(image_path: str, text_descriptions: list[str]) -> Optional[torch.Tensor]:
    """
    Compute similarity between image and text descriptions.
    Useful for image-grounded retrieval.
    
    Args:
        image_path: Path to chest X-ray
        text_descriptions: List of text descriptions to compare
        
    Returns:
        Similarity scores or None
    """
    model, preprocess, tokenizer = _load_biomedclip()
    
    if model is None or not os.path.exists(image_path):
        return None
    
    try:
        # Process image
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(_DEVICE)
        
        # Process text
        text_tokens = tokenizer(text_descriptions).to(_DEVICE)
        
        # Get features
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_tokens)
            
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity (cosine similarity via dot product)
            similarity = (image_features @ text_features.T).squeeze(0)
        
        return similarity.cpu()
    
    except Exception as e:
        print(f"ERROR: Failed to compute image-text similarity: {e}")
        return None


def run_vision_agent(state: MaraGraphState) -> Dict[str, Any]:
    """
    Main entry point for Vision Agent.
    Processes chest X-ray image if provided and adds embeddings to state.
    
    Args:
        state: Current pipeline state
        
    Returns:
        Updated state dict with image_embeddings and image_features
    """
    print("--- RUNNING VISION AGENT (VA) ---")
    
    image_path = (state.get("image_path") or "").strip()
    
    # Skip if no image provided
    if not image_path:
        print("VA: No image path provided, skipping vision processing.")
        return {
            "image_embeddings": None,
            "image_features": None,
        }
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"VA: Image file not found: {image_path}")
        return {
            "image_embeddings": None,
            "image_features": None,
        }
    
    # Extract embeddings
    embeddings = extract_image_embeddings(image_path)
    
    if embeddings is None:
        print("VA: Failed to extract image embeddings")
        return {
            "image_embeddings": None,
            "image_features": None,
        }
    
    print(f"VA: Successfully extracted image embeddings (shape: {embeddings.shape})")
    
    # Optionally compute similarity with retrieved docs for image-grounded retrieval
    retrieved_docs = state.get("retrieved_docs", [])
    image_text_scores = None
    
    if retrieved_docs:
        doc_texts = [doc.get("content", "") for doc in retrieved_docs]
        image_text_scores = get_image_text_similarity(image_path, doc_texts)
        
        if image_text_scores is not None:
            print(f"VA: Computed image-text similarity for {len(doc_texts)} documents")
            # You could re-rank retrieved docs based on image similarity here
    
    return {
        "image_embeddings": embeddings,
        "image_features": {
            "shape": list(embeddings.shape),
            "device": str(embeddings.device),
            "image_path": image_path,
        },
        "image_text_scores": image_text_scores,
    }
