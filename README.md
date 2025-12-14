# Brain MRI Classification Project

## Project Overview
This project implements a brain MRI image classification system to identify three critical brain conditions:
- **Hemorrhagic Stroke**: Brain bleeding caused by ruptured blood vessels
- **Ischemic Stroke**: Brain tissue damage due to blocked blood flow
- **Tumor**: Abnormal brain tissue growth (glioma, meningioma, and pituitary tumors)

## Project Structure
```
Stroke-Tumor-Classification/
├── READEME.md                    # This file - Main project documentation
└── data/
    ├── dataset_preparation.ipynb     # Step 1: Organize raw datasets
    ├── data_preprocessing.ipynb      # Step 2: Preprocess and augment images
    └── eda.ipynb                     # Step 3: Exploratory Data Analysis
├── requirements.txt              # Python dependencies
└── content/
    ├── README.md                 # Dataset documentation
    ├── dataset/                  # Organized dataset (after Step 1)
    │   ├── Hemorrhagic/
    │   ├── Ischemic/
    │   └── Tumor/
    └── processed_data/           # Preprocessed dataset (after Step 2)
        ├── Hemorrhagic/
        ├── Ischemic/
        └── Tumor/
```

## Setup Instructions

### Step 1: Clone/Download the Project
```bash
git clone https://github.com/yusufafify/Stroke-Tumor-Classification.git
cd project
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**Required packages include:**
- opencv-python
- numpy
- matplotlib
- seaborn
- torch & torchvision
- Pillow
- tqdm

### Step 3: Download Original Datasets

You need to download two datasets from Kaggle:

#### Dataset 1: Brain Tumor Classification
- **Link**: https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri
- **Download Instructions**:
  1. Go to the Kaggle link above
  2. Click "Download" button (requires Kaggle account)
  3. You'll get a file named `brain-tumor-classification-mri.zip`

#### Dataset 2: Brain Stroke MRI Images
- **Link**: https://www.kaggle.com/datasets/mitangshu11/brain-stroke-mri-images
- **Download Instructions**:
  1. Go to the Kaggle link above
  2. Click "Download" button (requires Kaggle account)
  3. You'll get a file named `brain-stroke-mri-images.zip`

### Step 4: Extract Datasets

Create the following directory structure and extract the datasets:

```
project/
└── content/
    ├── Brain_Tumor_Classification/
    │   ├── Training/           # Extract brain-tumor-classification-mri.zip here
    │   │   ├── glioma_tumor/
    │   │   ├── meningioma_tumor/
    │   │   ├── pituitary_tumor/
    │   │   └── no_tumor/
    │   └── Testing/
    └── Brain_Stroke_MRI/
        └── Dataset_MRI_Folder/ # Extract brain-stroke-mri-images.zip here
            ├── Haemorrhagic/   # Contains Sub_category_stroke folders with different MRI modalities
            └── Ischemic/       # Contains Sub_category_stroke folders with different MRI modalities
```


## Workflow - How to Run

### Step 1: Dataset Preparation
Open and run `dataset_preparation.ipynb`:

**What it does:**
- Recursively extracts images from nested Sub_category_stroke/modality folders
- Combines all tumor types (glioma, meningioma, pituitary) into single "Tumor" class
- Organizes images into clean structure: `content/dataset/[Hemorrhagic|Ischemic|Tumor]/`
- Removes temporary folders to save space

**Run all cells in order.**

**Expected output:**
```
content/dataset/
├── Hemorrhagic/  (~400-600 images)
├── Ischemic/     (~400-600 images)
└── Tumor/        (~2000-3000 images)
```

### Step 2: Exploratory Data Analysis (EDA)
Open and run `eda.ipynb`:

**What it does:**
- Visualizes random samples from each class
- Shows class distribution statistics
- Analyzes image dimensions
- Generates summary charts

**Run all cells in order.**

**Expected output:**
- Sample image grids per class
- Bar charts showing class balance
- Console statistics about dataset


### Step 3: Data Preprocessing & Augmentation
Open and run `data_preprocessing.ipynb`:

**What it does:**
- **ROI Extraction**: Removes black backgrounds and focuses on brain tissue
- **CLAHE Enhancement**: Improves contrast using adaptive histogram equalization
- **Resizing**: Standardizes all images to 224×224 pixels
- **Augmentation**: Balances dataset by generating augmented samples (rotation, flip, shift)
- **Target**: 2000 images per class for balanced training

**Run all cells in order.**

**Expected output:**
```
content/processed_data/
├── Hemorrhagic/  (~2000 images: real_*.jpg + aug_*.jpg)
├── Ischemic/     (~2000 images: real_*.jpg + aug_*.jpg)
└── Tumor/        (~2000 images: real_*.jpg)
```

**Visualization:**
The notebook includes validation plots showing:
- Raw vs processed images
- Augmentation examples
- Pixel intensity histograms
- Class distribution charts


## Troubleshooting

### Issue: "Source directory not found"
**Solution**: Double-check that you extracted the datasets to the correct paths:
- `content/Brain_Stroke_MRI/Dataset_MRI_Folder/`
- `content/Brain_Tumor_Classification/Training/`


### Issue: Out of memory during preprocessing
**Solution**: In `data_preprocessing.ipynb`, reduce `TARGET_COUNT` from 2000 to 1500 or 1000.

### Issue: OpenCV errors
**Solution**: 
```bash
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python
```

## File Descriptions

| File | Purpose |
|------|---------|
| `dataset_preparation.ipynb` | Organizes raw Kaggle datasets into unified structure |
| `data_preprocessing.ipynb` | Applies ROI extraction, CLAHE, augmentation, and balancing |
| `eda.ipynb` | Visualizes dataset statistics and sample images |
| `requirements.txt` | Lists all Python dependencies |
| `content/README.md` | Documentation for the organized dataset |

## Next Steps

After running all notebooks successfully, you'll have:
1. **Organized Dataset**: `content/dataset/` - Original images cleaned and organized
2. **Processed Dataset**: `content/processed_data/` - Augmented, balanced, and ready for training
3. **Understanding**: Clear visualizations from EDA

**Ready for Model Training!** 🚀

The processed dataset in `content/processed_data/` is now ready to be split into train/validation/test sets and used for training CNN models (ResNet, EfficientNet, etc.).
