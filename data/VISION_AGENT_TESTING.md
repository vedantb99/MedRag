# Vision Agent Testing

## Setup

The Vision Agent uses BiomedCLIP to process chest X-ray images and extract embeddings.

## Testing without a real chest X-ray

Since we don't have a real chest X-ray image in the repository, you can test the vision agent in two ways:

### Option 1: Download a sample chest X-ray

Download a sample chest X-ray from a public medical imaging dataset:
- NIH Chest X-ray Dataset: https://nihcc.app.box.com/v/ChestXray-NIHCC
- MIMIC-CXR (requires credentialed access): https://physionet.org/content/mimic-cxr/2.0.0/

### Option 2: Create a test image

For quick testing, you can use any image:

```bash
# Create a simple test image using Python
python -c "
from PIL import Image
import numpy as np

# Create a simple grayscale image (simulating an X-ray)
img = Image.fromarray(np.random.randint(0, 256, (512, 512), dtype=np.uint8), mode='L')
img.save('data/test_xray.png')
print('Test image created at data/test_xray.png')
"
```

### Option 3: Test with the current setup

The vision agent will gracefully skip processing if no valid image is provided:

```bash
# This will work - vision agent will skip
python main.py --clinical_query "Patient with chest pain"

# This will attempt to process the image
python main.py \
  --clinical_query "Patient with chest pain" \
  --image_path "data/test_xray.png"
```

## Expected Behavior

When a valid image is provided:
1. Vision Agent loads BiomedCLIP model
2. Extracts image embeddings
3. Optionally computes image-text similarity with retrieved documents
4. Embeddings are available for the Generation Agent to use

When no image or invalid path:
1. Vision Agent prints: "No image path provided, skipping vision processing."
2. Pipeline continues normally with text-only processing

## BiomedCLIP Model

The Vision Agent uses: `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`

This model is specifically trained on biomedical images and text pairs, making it ideal for radiology applications.

## Future Enhancements

- Image-guided retrieval: Re-rank retrieved documents using image-text similarity
- Multimodal generation: Incorporate image features directly into the LLM prompt
- Visual grounding: Identify which regions of the image correspond to generated findings
