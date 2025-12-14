# Brain MRI Classification Dataset

## Overview
This dataset contains brain MRI images organized for classification of three critical conditions:
- **Hemorrhagic Stroke**: Brain bleeding caused by ruptured blood vessels
- **Ischemic Stroke**: Brain tissue damage due to blocked blood flow
- **Tumor**: Abnormal brain tissue growth (includes glioma, meningioma, and pituitary tumors)

## Current Dataset Structure

After running the complete preprocessing pipeline, you'll have:

```
content/
├── README.md                    # This file
├── dataset/                     # Organized raw dataset (after dataset_preparation.ipynb)
│   ├── Hemorrhagic/            # ~400-600 hemorrhagic stroke images
│   ├── Ischemic/               # ~400-600 ischemic stroke images
│   └── Tumor/                  # ~2000-3000 tumor images (glioma + meningioma + pituitary)
│
└── processed_data/              # Preprocessed dataset (after data_preprocessing.ipynb)
    ├── Hemorrhagic/            # ~2000 images (real + augmented)
    │   ├── real_*.jpg          # Original preprocessed images
    │   └── aug_*.jpg           # Augmented images for class balance
    ├── Ischemic/               # ~2000 images (real + augmented)
    │   ├── real_*.jpg
    │   └── aug_*.jpg
    └── Tumor/                  # ~2000 images (real + augmented)
        └── real_*.jpg
```

## Data Sources

This dataset was compiled from two Kaggle datasets:

### 1. Brain Stroke MRI Dataset
- **Source**: https://www.kaggle.com/datasets/mitangshu11/brain-stroke-mri-images
- **Content**: Multiple MRI modalities from stroke patients
  - DWI (Diffusion-Weighted Imaging)
  - GRE (Gradient Echo)
  - SWI (Susceptibility-Weighted Imaging)
  - T2 sequences
  - T2-FLAIR
- **Structure**: Nested folders by patient ID and MRI modality
- **Classes Used**: Haemorrhagic → Hemorrhagic, Ischemic → Ischemic

### 2. Brain Tumor Classification Dataset
- **Source**: https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri
- **Content**: Brain tumor MRI images from multiple tumor types
- **Classes Used**: 
  - glioma_tumor → Tumor
  - meningioma_tumor → Tumor
  - pituitary_tumor → Tumor
  - no_tumor → (excluded)

## How the Dataset Was Created

### Stage 1: Organization (`dataset_preparation.ipynb`)
The raw Kaggle datasets are extracted and organized into a clean three-class structure:

**Input:**
```
content/
├── Brain_Stroke_MRI/
│   └── Dataset_MRI_Folder/
│       ├── Haemorrhagic/
│       │   ├── Kuppusamy_Stroke_Haemorrhagic/
│       │   │   ├── DWI/
│       │   │   ├── GRE/
│       │   │   └── ...
│       │   └── Kuppusamy_Stroke_Haemorrhagic/
│       └── Ischemic/
│           └── (similar structure)
└── Brain_Tumor_Classification/
    └── Training/
        ├── glioma_tumor/
        ├── meningioma_tumor/
        ├── pituitary_tumor/
        └── no_tumor/
```

**Output:**
```
content/dataset/
├── Hemorrhagic/
├── Ischemic/
└── Tumor/
```

**Process:**
- Recursive extraction from nested Stroke_sub_category/modality folders
- Filename preservation with Stroke_sub_category/modality identifiers
- All three tumor types merged into single "Tumor" class
- Removes temporary folders to save disk space

### Stage 2: Preprocessing (`data_preprocessing.ipynb`)
Raw images are cleaned, enhanced, and augmented for model training:

**Preprocessing Pipeline:**
1. **ROI Extraction**: Removes black backgrounds using Otsu thresholding and contour detection
2. **Masking**: Isolates brain tissue with 20-pixel padding
3. **Resizing**: Standardizes to 224×224 pixels (standard CNN input)
4. **CLAHE Enhancement**: Adaptive contrast enhancement for better feature visibility
5. **Normalization**: Pixel values normalized to [0, 255]

**Augmentation (for class balancing):**
- Random rotation (±10°)
- Horizontal flip (50% probability)
- Random affine transform (translation ±5%, scale 0.95-1.05x)
- **Target**: 2000 images per class

**File Naming Convention:**
- `real_*.jpg`: Original preprocessed images
- `aug_*.jpg`: Augmented images generated for class balance

## Image Formats and Properties

**Format**: JPEG (.jpg)
**Resolution**: 224×224 pixels (post-processing)
**Color Space**: RGB (converted from BGR after OpenCV processing)
**Bit Depth**: 8-bit per channel

**Original Formats** (in raw Kaggle datasets):
- `.jpg`, `.jpeg`, `.png`
- Variable resolutions (typically 256×256 to 512×512)

## Dataset Statistics

### Class Distribution (after preprocessing)
| Class | Real Images | Augmented Images | Total |
|-------|------------|------------------|-------|
| Hemorrhagic | ~400-600 | ~1400-1600 | ~2000 |
| Ischemic | ~400-600 | ~1400-1600 | ~2000 |
| Tumor | ~2000-3000 | ~0-1000 | ~2000 |

**Total Dataset Size**: ~6000 images (balanced for training)

### Image Characteristics
- **Average pixel intensity**: Normalized across all images
- **Background**: Pure black (0, 0, 0) due to ROI masking
- **Contrast**: Enhanced using CLAHE for better feature distinction
- **Artifacts**: Minimized through ROI extraction and quality filtering