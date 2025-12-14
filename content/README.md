# Brain MRI Classification Dataset

## Overview
This dataset contains brain MRI images organized for classification of three critical conditions:
- **Hemorrhagic Stroke**: Brain bleeding caused by ruptured blood vessels
- **Ischemic Stroke**: Brain tissue damage due to blocked blood flow
- **Tumor**: Abnormal brain tissue growth (includes glioma, meningioma, and pituitary tumors)

## Dataset Structure
```
dataset/
├── Hemorrhagic/    # Hemorrhagic stroke MRI images
├── Ischemic/       # Ischemic stroke MRI images
└── Tumor/          # Brain tumor MRI images
```

## Data Sources
This dataset was compiled from two primary sources:
1. **Brain Stroke MRI Dataset**: Contains various MRI modalities (DWI, GRE, SWI, T2, T2-FLAIR) from multiple patients with hemorrhagic and ischemic strokes
2. **Brain Tumor Classification Dataset**: Contains tumor images including glioma, meningioma, and pituitary tumor cases

## Image Formats
- Supported formats: `.jpg`, `.jpeg`, `.png`
- Images are from various MRI sequences providing different tissue contrasts

## Usage
This dataset is prepared for training deep learning models for brain condition classification. The three-class structure makes it suitable for:
- Multi-class classification tasks
- Transfer learning experiments
- Medical image analysis research

## Preparation
Images were preprocessed and organized using `dataset_preparation.ipynb`, which:
- Recursively extracted images from nested patient and modality folders
- Consolidated multiple tumor types into a single Tumor class
- Created unique filenames to prevent overwrites
- Organized all images into a flat structure per class

## Notes
- Patient identifiers are preserved in filenames for traceability
- Different MRI modalities are included to provide comprehensive brain imaging data
- Dataset is balanced as much as possible across the three classes

---