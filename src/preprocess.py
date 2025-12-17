import numpy as np
import cv2
import warnings

warnings.filterwarnings('ignore')


def extract_roi_and_mask(image):
    """
    Robust ROI extraction for MRI:
    1. Blurs and Thresholds to find brain structure.
    2. Dilates to connect fragmented parts (like cerebellum).
    3. Crops the largest contour with padding.
    
    Args:
        image: OpenCV image (BGR format)
        
    Returns:
        ROI extracted image
    """
    img_copy = image.copy()
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    
    # 1. Blur to remove fine noise (text)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # 2. Otsu Thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Dilate to connect brain parts (Fixes the "fragmented brain" issue)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=3)
    
    # 4. Find Contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image  # Fallback

    # Find largest contour (The Brain)
    c = max(contours, key=cv2.contourArea)
    
    # Area Check: Ignore if the "brain" is tiny (likely noise)
    img_area = img_copy.shape[0] * img_copy.shape[1]
    if cv2.contourArea(c) < (0.05 * img_area):
        return image
    
    # 5. Masking (Black out the background)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [c], -1, 255, -1)
    masked_image = cv2.bitwise_and(img_copy, img_copy, mask=mask)
    
    # 6. Cropping with Padding
    x, y, w, h = cv2.boundingRect(c)
    pad = 40  # Generous padding to verify we don't cut off lesions
    
    x_new = max(0, x - pad)
    y_new = max(0, y - pad)
    w_new = min(img_copy.shape[1] - x_new, w + 2*pad)
    h_new = min(img_copy.shape[0] - y_new, h + 2*pad)
    
    roi = masked_image[y_new:y_new+h_new, x_new:x_new+w_new]
    return roi



def normalize_and_clahe(image):
    """
    Standardizes MRI intensity using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        image: OpenCV image (BGR format)
        
    Returns:
        Enhanced image with normalized intensity
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Normalize to 0-255 range
    final_norm = cv2.normalize(final, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return final_norm


def preprocess_mri_image(image):
    """
    Complete preprocessing pipeline for MRI images.
    Combines ROI extraction and CLAHE normalization.
    
    Args:
        image: OpenCV image (BGR format)
        
    Returns:
        Preprocessed image ready for model input
    """
    roi = extract_roi_and_mask(image)
    enhanced = normalize_and_clahe(roi)
    return enhanced
