# Brain MRI Classification & Segmentation Project

## Project Overview
This project implements a complete deep learning pipeline for brain MRI analysis, combining both classification and segmentation tasks to identify and localize three critical brain conditions:
- **Hemorrhagic Stroke**: Brain bleeding caused by ruptured blood vessels
- **Ischemic Stroke**: Brain tissue damage due to blocked blood flow
- **Tumor**: Abnormal brain tissue growth (glioma, meningioma, and pituitary tumors)

### Key Features
- 🔍 **Classification**: ResNet50-based model with 5-fold cross-validation and ensemble learning
- 🎯 **Segmentation**: U-Net architecture for lesion localization using pseudo-mask generation
- 🖼️ **Preprocessing Pipeline**: ROI extraction, CLAHE enhancement, and augmentation
- 📊 **Visualization**: Grad-CAM for classification interpretability and segmentation overlays
- 🚀 **Production-Ready Inference**: Modular classes for easy backend integration (Flask/FastAPI)

## Project Structure
```
Stroke-Tumor-Classification/
├── README.md                         # This file - Main project documentation
├── requirements.txt                  # Python dependencies
│
├── data/                             # Data preparation notebooks
│   ├── dataset_preparation.ipynb     # Step 1: Organize raw datasets
│   ├── data_preprocessing.ipynb      # Step 2: Preprocess and augment images
│   └── eda.ipynb                     # Step 3: Exploratory Data Analysis
│
├── model_notebooks/                  # Model training notebooks
│   ├── classification_model.ipynb    # ResNet50 training with cross-validation
│   └── segmentation_model.ipynb      # U-Net training with pseudo-masks
│
├── models/                           # Trained model weights
│   ├── classification_model.pth      # Best classification model (ensemble)
│   └── segmentation_model.pth        # Best segmentation model
│
├── src/                              # Production inference code
│   ├── inference.py                  # Main inference classes
│   └── preprocess.py                 # MRI preprocessing functions
│
└── content/                          # Datasets
    ├── README.md                     # Dataset documentation
    ├── dataset/                      # Organized dataset (after Step 1)
    │   ├── Hemorrhagic/
    │   ├── Ischemic/
    │   └── Tumor/
    └── processed_data/               # Preprocessed dataset (after Step 2)
        ├── Hemorrhagic/
        ├── Ischemic/
        └── Tumor/
```

## Setup Instructions

### Step 1: Clone/Download the Project
```bash
git clone https://github.com/yusufafify/Stroke-Tumor-Classification.git
cd Stroke-Tumor-Classification
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
- scikit-learn
- scipy

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


## Complete Workflow - How to Run

The project consists of 5 main stages: Dataset Preparation → Preprocessing → Model Training → Evaluation → Inference

---

### Stage 1: Dataset Preparation
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
├── Isageemic/     (~400-600 images)
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
- Bar agearts showing class balance
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

---

### Stage 4: Model Training

#### 4.1 Classification Model Training
Open and run `model_notebooks/classification_model.ipynb`:

**What it does:**
- **Architecture**: ResNet50 (pretrained on ImageNet)
- **Training Strategy**: 5-Fold Group Cross-Validation (patient-level splits)
- **Class Balancing**: WeightedRandomSampler to handle class imbalance
- **Data Augmentation**: Random flips, rotation, color jitter
- **Ensemble**: Averages weights from all 5 folds into final model
- **Progress Tracking**: tqdm progress bars for training monitoring

**Model Features:**
- Group K-Fold to prevent data leakage (same patient images stay in same fold)
- Balanced sampling ensures equal representation during training
- Early stopping saves best model per fold
- Final & Directory Descriptions

### Data Notebooks (`data/`)
| File | Purpose |
|------|---------|
| `dataset_preparation.ipynb` | Organizes raw Kaggle datasets into unified structure |
| `data_preprocessing.ipynb` | Applies ROI extraction, CLAHE, augmentation, and balancing |
| `eda.ipynb` | Visualizes dataset statistics and sample images |

### Model Notebooks (`model_notebooks/`)
| File | Purpose |
|------|---------|
| `classification_model.ipynb` | ResNet50 training with 5-fold CV, evaluation metrics, and Grad-CAM |
| `segmentation_model.ipynb` | U-Net training with pseudo-mask generation and visualization |

### Source Code (`src/`)
| File | Purpose |
|------|---------|
| `inference.py` | Production inference classes (Classification, Segmentation, Grad-CAM) |
| `preprocess.py` | MRI preprocessing functions (ROI extraction, CLAHE) |

### Models (`models/`)
| File | Purpose |
|------|---------|
| `classification_model.pth` | Trained ResNet50 ensemble weights (~94-95% accuracy) |
| `segmentation_model.pth` | Trained U-Net weights for lesion segmentation |

### Content (`content/`)
| Directory | Purpose |
|-----------|---------|
| `dataset/` | Organized raw images by class |
| `processed_data/` | Preprocessed, augmented, balanced images ready for training |

## Model Performance

### Classification Model
- **Architecture**: ResNet50 (pretrained)
- **Training**: 5-Fold Group Cross-Validation
- **Accuracy**: ~94-95% (ensemble average)
- **Classes**: Hemorrhagic, Ischemic, Tumor
- **Metrics**: 
  - High precision and recall across all classes
  - AUC scores > 0.95 for ROC curves
  - Balanced performance with class weighting

### Segmentation Model
- **Architecture**: U-Net
- **Training**: Self-supervised with pseudo-masks
- **Output**: Binary lesion masks
- **Lesion Detection**: Automatically highlights affected brain regions
- **Method**: K-Means clustering-based mask generation

## Technical Highlights

### Preprocessing Pipeline
1. **ROI Extraction**: Removes background, isolates brain tissue
2. **CLAHE Enhancement**: Improves contrast for better feature visibility
3. **Augmentation**: Rotation, flipping, color jittering for robustness
4. **Normalization**: ImageNet statistics for transfer learning

### Training Strategies
- **Group K-Fold**: Prevents data leakage by keeping patient scans together
- **Weighted Sampling**: Addresses class imbalance
- **Ensemble Learning**: Averages multiple fold models for stability
- **Early Stopping**: Saves best validation performance per fold

### Interpretability
- **Grad-CAM**: Visualizes which brain regions influence classification
- **Segmentation Overlays**: Shows exact lesion locations
- **Probability Maps**: Displays confidence across entire image

## Next Steps & Extensions

### Potential Improvements
1. **Data**: Collect more diverse datasets, especially for minority classes
2. **Models**: 
   - Try EfficientNet, Vision Transformers (ViT)
   - Experiment with attention mechanisms
   - Multi-task learning (joint classification + segmentation)
3. **Segmentation**: Obtain manual annotations for supervised training
4. **Deployment**: Create REST API with FastAPI/Flask
5. **Clinical Integration**: Add DICOM support, patient metadata handling

### Deployment Options
- **Web Application**: Flask/FastAPI + React frontend
- **Desktop Application**: PyQt/Tkinter GUI
- **Cloud Deployment**: AWS Lambda, Google Cloud Run, Azure Functions
- **Edge Deployment**: ONNX conversion for mobile/embedded devices

## Citation & Acknowledgments

### Datasets Used
1. **Brain Tumor Classification MRI Dataset**
   - Source: [Kaggle - Brain Tumor Classification](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
   
2. **Brain Stroke MRI Images**
   - Source: [Kaggle - Brain Stroke MRI Images](https://www.kaggle.com/datasets/mitangshu11/brain-stroke-mri-images)

### Technologies
- **PyTorch**: Deep learning framework
- **OpenCV**: Image preprocessing
- **scikit-learn**: Cross-validation and metrics
- **ResNet50**: Classification backbone (He et al., 2015)
- **U-Net**: Segmentation architecture (Ronneberger et al., 2015)

## License

This project is for educational and research purposes. Please ensure proper citations when using the datasets or adapting the code.

---

**Project Status**: ✅ Complete Pipeline - Ready for Deployment

For questions or contributions, please open an issue on the repository
#### 4.2 Segmentation Model Training
Open and run `model_notebooks/segmentation_model.ipynb`:

**What it does:**
- **Architecture**: U-Net with skip connections
- **Training Strategy**: Self-supervised using pseudo-mask generation
- **Pseudo-Mask Generation**: K-Means clustering to automatically identify lesions
  - Hemorrhagic & Tumor: Focus on brightest clusters (hyperdense regions)
  - Ischemic: Focus on darkest clusters (hypodense regions)
- **Loss Function**: Binary Cross-Entropy (BCE)
- **Output**: Binary masks highlighting lesion locations

**How Pseudo-Masks Work:**
Since we don't have manual segmentation labels, the model uses intelligent clustering:
1. Convert MRI to grayscale
2. Apply K-Means (k=3) to separate dark/mid/bright regions
3. Select appropriate cluster based on pathology type
4. Generate binary mask for training

**Run all cells in order.**

**Expected output:**
```
Epoch 1 Segmentation Loss: 0.xxxx
...
✅ Model saved as: unet_lesion_model_new.pth
```

**Visualization:**
- Side-by-side comparison: Original | Probability Map | Overlay
- Shows lesion localization accuracy
- Red overlay highlights detected regions

---

### Stage 5: Inference & Deployment

The `src/` directory contains production-ready inference code for backend integration.

#### Using the Inference System

**Test the inference pipeline:**
```bash
python src/inference.py
```

**Available Inference Classes:**

Each class is standalone and performs a specific task:

#### 1. **ClassificationInference** - Brain condition classification
Preprocesses the image and returns the diagnosis with confidence scores.

```python
from src.inference import ClassificationInference

classifier = ClassificationInference("models/classification_model.pth")
result = classifier.predict("brain_scan.jpg")

print(f"Diagnosis: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"All probabilities: {result['probabilities']}")
```

**Output:**
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

#### 2. **SegmentationInference** - Lesion segmentation
Preprocesses the image and returns binary mask highlighting the affected region.

```python
from src.inference import SegmentationInference

segmenter = SegmentationInference("models/segmentation_model.pth")
result = segmenter.predict("brain_scan.jpg", threshold=0.5, return_overlay=True)

print(f"Lesion area: {result['lesion_area_ratio']:.2%}")
# Access: result['mask'], result['probability_map'], result['overlay']
```

**Output:**
```python
{
    'mask': np.array(...),              # Binary mask (0s and 1s)
    'probability_map': np.array(...),   # Raw probabilities (0-1)
    'lesion_area_ratio': 0.15,         # 15% of brain affected
    'lesion_pixels': 7500,
    'total_pixels': 50176,
    'overlay': np.array(...)            # RGB image with red overlay
}
```

#### 3. **GradCAMInference** - Interpretability visualization
Generates Grad-CAM heatmap showing which brain regions influenced the classification decision.

```python
from src.inference import GradCAMInference

gradcam = GradCAMInference("models/classification_model.pth")
result = gradcam.generate_heatmap_overlay("brain_scan.jpg", alpha=0.4)

print(f"Diagnosis: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
# Access: result['heatmap'], result['overlay']
```

**Output:**
```python
{
    'predicted_class': 'Tumor',
    'predicted_index': 2,
    'confidence': 0.96,
    'probabilities': {...},
    'heatmap': np.array(...),    # Normalized heatmap (0-1)
    'overlay': np.array(...)     # Image with colored heatmap overlay
}
```

---

**Key Features:**
- ✅ **Standalone classes** - Each class performs a single, focused task
- ✅ **Automatic preprocessing** - ROI extraction and CLAHE applied automatically
- ✅ **Flexible input** - Supports file paths, PIL Images, and numpy arrays
- ✅ **GPU acceleration** - Auto-detects and uses CUDA when available
- ✅ **Production-ready** - Clean API for Flask/FastAPI integration


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
