# Source Code Documentation

This directory contains production-ready inference code for the Brain MRI Classification & Segmentation project.

## Files Overview

### 📄 `preprocess.py`
Contains MRI-specific preprocessing functions used by all inference models.

#### Functions:

**`extract_roi_and_mask(image)`**
- **Purpose**: Extracts the brain region of interest (ROI) from MRI scans
- **What it does**:
  - Removes black backgrounds and text artifacts
  - Identifies brain tissue using thresholding and morphological operations
  - Crops to the brain region with padding to avoid cutting off lesions
- **Input**: OpenCV image (BGR format)
- **Output**: Cropped and masked brain image
- **Why it's important**: Focuses the model on relevant brain tissue, removing distractions

**`normalize_and_clahe(image)`**
- **Purpose**: Enhances MRI image contrast for better feature visibility
- **What it does**:
  - Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Normalizes pixel intensity to a standard range
- **Input**: OpenCV image (BGR format)
- **Output**: Contrast-enhanced image
- **Why it's important**: Makes lesions and brain structures more visible to the model

**`preprocess_mri_image(image)`**
- **Purpose**: Complete preprocessing pipeline combining both functions above
- **What it does**:
  1. Extracts ROI (removes background)
  2. Applies CLAHE enhancement
- **Input**: Raw MRI image (BGR format)
- **Output**: Preprocessed image ready for model input
- **Usage**: This is the main function called by all inference classes

---

### 📄 `inference.py`
Contains three standalone inference classes for different tasks.

## Inference Classes

### 1. 🔍 `ClassificationInference`

**Purpose**: Classifies brain MRI scans into three categories: Hemorrhagic, Ischemic, or Tumor

#### Key Methods:

**`__init__(model_path, device=None)`**
- Initializes the ResNet50 classification model
- `model_path`: Path to the trained `.pth` model file
- `device`: 'cuda' for GPU, 'cpu' for CPU, or None for auto-detect

**`predict(image)`**
- **What it does**:
  1. Preprocesses the input image (ROI extraction + CLAHE)
  2. Runs it through the ResNet50 model
  3. Returns diagnosis with confidence scores
- **Input**: Image (file path, PIL Image, or numpy array)
- **Output**: Dictionary with:
  - `predicted_class`: Name of the diagnosis
  - `confidence`: Confidence score (0-1)
  - `probabilities`: Probability for each class

**Example Usage:**
```python
from inference import ClassificationInference

# Initialize
classifier = ClassificationInference("../models/classification_model.pth")

# Predict
result = classifier.predict("brain_scan.jpg")
print(f"Diagnosis: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

---

### 2. 🎯 `SegmentationInference`

**Purpose**: Identifies and highlights the exact location of brain lesions

#### Key Methods:

**`__init__(model_path, device=None)`**
- Initializes the U-Net segmentation model
- `model_path`: Path to the trained `.pth` model file

**`predict(image, threshold=0.5, return_overlay=False)`**
- **What it does**:
  1. Preprocesses the input image
  2. Generates a probability map of lesion locations
  3. Creates a binary mask (0s and 1s)
  4. Optionally creates a red overlay visualization
- **Input**: 
  - `image`: Image to segment
  - `threshold`: Probability threshold for binary mask (default: 0.5)
  - `return_overlay`: Whether to return visualization
- **Output**: Dictionary with:
  - `mask`: Binary mask (numpy array)
  - `probability_map`: Raw probabilities (0-1)
  - `lesion_area_ratio`: Percentage of brain affected
  - `overlay`: (Optional) Image with red lesion overlay

**Example Usage:**
```python
from inference import SegmentationInference

# Initialize
segmenter = SegmentationInference("../models/segmentation_model.pth")

# Predict
result = segmenter.predict("brain_scan.jpg", return_overlay=True)
print(f"Lesion covers {result['lesion_area_ratio']:.1%} of brain")

# Access the mask
mask = result['mask']  # Binary mask as numpy array
overlay = result['overlay']  # Visualization
```

---

### 3. 🔥 `GradCAMInference`

**Purpose**: Shows which brain regions the model focused on when making its classification decision (explainable AI)

#### Key Methods:

**`__init__(model_path, device=None)`**
- Initializes the model with Grad-CAM hooks for visualization
- Uses the same ResNet50 architecture as classification

**`generate_heatmap_overlay(image, target_class=None, alpha=0.4)`**
- **What it does**:
  1. Runs classification
  2. Generates a Grad-CAM heatmap showing important regions
  3. Overlays the heatmap on the original image
- **Input**:
  - `image`: Image to analyze
  - `target_class`: Which class to explain (None = predicted class)
  - `alpha`: Heatmap transparency (0-1)
- **Output**: Dictionary with:
  - Classification results (class, confidence, probabilities)
  - `heatmap`: Raw Grad-CAM heatmap
  - `overlay`: Image with colored heatmap overlay

**Example Usage:**
```python
from inference import GradCAMInference

# Initialize
gradcam = GradCAMInference("../models/classification_model.pth")

# Generate explanation
result = gradcam.generate_heatmap_overlay("brain_scan.jpg")
print(f"Model looked at: (see heatmap)")
print(f"Diagnosis: {result['predicted_class']}")

# Save visualization
import cv2
cv2.imwrite("explanation.jpg", cv2.cvtColor(result['overlay'], cv2.COLOR_RGB2BGR))
```

---

## Quick Start Guide

### Prerequisites
```bash
pip install torch torchvision opencv-python pillow numpy
```

### Basic Usage - All Three Models

```python
from inference import ClassificationInference, SegmentationInference, GradCAMInference

# Initialize all models
classifier = ClassificationInference("../models/classification_model.pth")
segmenter = SegmentationInference("../models/segmentation_model.pth")
gradcam = GradCAMInference("../models/classification_model.pth")

# Analyze an image
image_path = "path/to/brain_scan.jpg"

# 1. Get diagnosis
diagnosis = classifier.predict(image_path)
print(f"Diagnosis: {diagnosis['predicted_class']} ({diagnosis['confidence']:.2%})")

# 2. Find lesion location
segmentation = segmenter.predict(image_path, return_overlay=True)
print(f"Lesion area: {segmentation['lesion_area_ratio']:.1%}")

# 3. Explain the decision
explanation = gradcam.generate_heatmap_overlay(image_path)
print("Heatmap shows which regions influenced the diagnosis")
```

---

## Input Formats

All inference classes accept **four** image formats:

1. **File path (string)**
   ```python
   result = classifier.predict("scan.jpg")
   ```

2. **PIL Image**
   ```python
   from PIL import Image
   img = Image.open("scan.jpg")
   result = classifier.predict(img)
   ```

3. **NumPy array**
   ```python
   import cv2
   img = cv2.imread("scan.jpg")
   result = classifier.predict(img)
   ```

4. **Base64 encoded string** ⭐ (For API/Frontend usage)
   ```python
   # With data URI prefix
   base64_str = "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
   result = classifier.predict(base64_str)
   
   # Or without prefix
   base64_str = "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBD..."
   result = classifier.predict(base64_str)
   ```

---

## Preprocessing Pipeline

All models automatically apply the same preprocessing:

```
Raw MRI Image
    ↓
ROI Extraction (remove background, crop brain)
    ↓
CLAHE Enhancement (improve contrast)
    ↓
Resize to 224×224
    ↓
Normalize with ImageNet statistics
    ↓
Ready for Model
```

**You don't need to manually preprocess images** - it's done automatically!

---

## Output Formats

### Classification Output
```python
{
    'predicted_class': 'Hemorrhagic',
    'predicted_index': 0,
    'confidence': 0.94,
    'probabilities': {
        'Hemorrhagic': 0.94,
        'Ischemic': 0.04,
        'Tumor': 0.02
    }
}
```

### Segmentation Output
```python
{
    'mask': numpy.ndarray (H×W),           # Binary: 0 = background, 1 = lesion
    'probability_map': numpy.ndarray (H×W), # Float: 0-1 probabilities
    'lesion_area_ratio': 0.15,             # 15% of brain affected
    'lesion_pixels': 7500,
    'total_pixels': 50176,
    'overlay': numpy.ndarray (H×W×3)       # RGB image with red overlay
}
```

### Grad-CAM Output
```python
{
    'predicted_class': 'Tumor',
    'predicted_index': 2,
    'confidence': 0.96,
    'probabilities': {...},
    'heatmap': numpy.ndarray (H×W),        # Normalized 0-1
    'overlay': numpy.ndarray (H×W×3)       # RGB with colored heatmap
}
```

---

## Questions

**Q: Do I need to preprocess images before passing them?**  
A: No! All preprocessing is automatic. Just pass the raw image.

**Q: What format should images be in?**  
A: Any of: file path string, PIL Image, or numpy array (BGR/RGB).

**Q: Can I use CPU instead of GPU?**  
A: Yes! Pass `device='cpu'` when initializing, or it auto-detects.

**Q: How do I save the overlay images?**  
A: Use OpenCV:
```python
import cv2
result = segmenter.predict(img, return_overlay=True)
cv2.imwrite("output.jpg", cv2.cvtColor(result['overlay'], cv2.COLOR_RGB2BGR))
```

**Q: Why three separate classes instead of one combined class?**  
A: Modularity! Use only what you need. Each class has a single, focused responsibility.

---

## Troubleshooting

**Issue**: `FileNotFoundError: Model file not found`  
**Solution**: Check that model paths are correct relative to where you're running the script.

**Issue**: `CUDA out of memory`  
**Solution**: Use `device='cpu'` when initializing, or process images one at a time.

---
