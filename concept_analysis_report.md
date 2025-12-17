# Dataset-Specific Concept Analysis Report
## Label-Free Concept Bottleneck Model (LF-CBM)

### Date: December 17, 2025

---

## Dataset Overview

- **Hemorrhagic**: 83 images (77% SWI, 14% DWI, 8% GRE)
- **Ischemic**: 113 images (Mixed sequences, 17% DWI)
- **Tumor**: 115 images (Various contrasts)

---

## Visual Characteristics Analysis

### Quantitative Metrics

| Class | Avg Brightness | Avg Contrast | Edge Density | Key Observation |
|-------|----------------|--------------|--------------|-----------------|
| **Hemorrhagic** | 0.258 ± 0.050 | 0.313 ± 0.026 | 0.0362 ± 0.0061 | Moderate brightness, dark focal regions |
| **Ischemic** | 0.194 ± 0.041 | 0.225 ± 0.045 | 0.0208 ± 0.0080 | Darkest class, low contrast |
| **Tumor** | 0.301 ± 0.058 | 0.280 ± 0.034 | 0.0414 ± 0.0097 | Brightest, highest edge density |

### MRI Sequence Distribution

**Hemorrhagic:**
- SWI (Susceptibility-Weighted Imaging): 77.1%
- DWI (Diffusion-Weighted Imaging): 14.5%
- GRE (Gradient Echo): 8.4%

**Ischemic:**
- Mixed/Unlabeled: 73.5%
- DWI: 16.8%
- T2/FLAIR: ~10%

**Tumor:**
- Various sequences: 100%

---

## Concept Bank Comparison

### ❌ Original Generic Concepts (Not Dataset-Specific)

**Tumor:**
- brain tumor
- mass effect
- midline shift
- ring-enhancing lesion
- edema surrounding mass
- compressed ventricles

**Hemorrhagic:**
- intracranial hemorrhage
- hyperdense blood
- white blood pool
- hematoma

**Ischemic:**
- ischemic stroke
- hypodense tissue
- vascular territory infarct
- dark tissue area

### ✅ New Dataset-Specific Concepts (Based on Actual Images)

**Tumor (9 concepts):**
1. space-occupying lesion
2. mass effect present
3. irregular lesion margins
4. vasogenic edema pattern
5. central hypointensity
6. heterogeneous signal intensity
7. fingerlike edema extension
8. structural displacement
9. necrotic center

**Hemorrhagic (8 concepts):**
1. dark signal on susceptibility weighted imaging
2. blooming artifact present
3. focal hypointense region
4. blood products signal intensity
5. susceptibility artifact
6. well-circumscribed hemorrhagic lesion
7. acute bleeding pattern
8. hypointense on SWI

**Ischemic (8 concepts):**
1. bright signal on diffusion weighted imaging
2. territorial distribution pattern
3. vascular territory involvement
4. diffusion restriction present
5. wedge-shaped lesion
6. cortical gray matter involvement
7. hypointense region on apparent diffusion coefficient
8. acute infarct pattern

**Normal/Control (5 concepts):**
1. normal gray matter signal
2. normal white matter signal
3. symmetric brain structures
4. no abnormal signal intensity
5. preserved brain anatomy

---

## Key Improvements

### 1. **MRI Sequence-Specific Language**
- **Before**: "hyperdense blood" (CT terminology)
- **After**: "dark signal on susceptibility weighted imaging" (MRI-specific)

### 2. **Precise Signal Characteristics**
- **Before**: "dark tissue area" (vague)
- **After**: "hypointense region on apparent diffusion coefficient" (precise)

### 3. **Pattern Recognition**
- **Added**: "territorial distribution pattern" for ischemic strokes
- **Added**: "blooming artifact present" for hemorrhages on SWI
- **Added**: "fingerlike edema extension" for tumor vasogenic edema

### 4. **Sequence-Aware Descriptions**
- Hemorrhagic concepts emphasize SWI characteristics (77% of data)
- Ischemic concepts emphasize DWI/ADC patterns
- Tumor concepts focus on structural changes visible across sequences

### 5. **Anatomical Precision**
- **Added**: "cortical gray matter involvement"
- **Added**: "structural displacement"
- **Added**: "well-circumscribed hemorrhagic lesion"

---

## Expected Benefits

1. **Higher Accuracy**: Concepts match actual image characteristics
2. **Better Explainability**: Terms align with radiological observations
3. **Sequence-Aware**: Concepts reflect dominant MRI sequences in dataset
4. **Clinical Relevance**: Uses standard neuroradiology terminology
5. **Reduced Confusion**: Avoids CT terminology when working with MRI data

---

## Recommendations

### For Training:
- Use these dataset-specific concepts in the LF-CBM model
- Monitor which concepts have highest weights per class
- Consider adding/removing concepts based on model performance

### For Validation:
- Check if high-weight concepts align with radiological expectations
- Validate explanations with domain experts
- Compare performance against generic concepts

### For Future Work:
- If dataset changes (e.g., adds CT scans), update concepts accordingly
- Consider creating separate concept banks for different MRI sequences
- Explore adding temporal concepts (acute vs chronic phases)

---

## Files Generated

1. `dataset_samples.png` - Visual samples from each class
2. `detailed_comparison.png` - Side-by-side comparison with sequence labels
3. `analyze_dataset.py` - Dataset analysis script
4. `detailed_visual_analysis.py` - Detailed visual pattern analysis
5. `cbm_model.ipynb` - Updated notebook with new concepts

---

## Total Concepts: 30
- Tumor: 9 concepts
- Hemorrhagic: 8 concepts
- Ischemic: 8 concepts
- Normal: 5 concepts

This provides a rich feature space for the LF-CBM model while maintaining interpretability.
