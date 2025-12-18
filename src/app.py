"""
Flask Backend Server for Brain MRI Classification & Segmentation
Provides a single unified API endpoint for the complete workflow:
1. Upload MRI image
2. Preprocess (ROI extraction + CLAHE)
3. Classify (Hemorrhagic/Ischemic/Tumor)
4. Segment lesion
5. Generate Grad-CAM visualization
"""

import os
import sys
import base64
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image

from inference import ClassificationInference, SegmentationInference, GradCAMInference, LFCBMInference, ResUNetSegmentationInference
# from lfcbm_inference import LFCBMInference

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Configuration
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
CLASSIFICATION_MODEL_PATH = os.path.join(MODELS_DIR, 'lfcbm_model.pth') 
RESNET_MODEL_PATH = os.path.join(MODELS_DIR, 'classification_model.pth')
SEGMENTATION_MODEL_PATH = os.path.join(MODELS_DIR, 'resunet_segmentation_model.pth')  # ResUNet model

# Global model instances (lazy loading)
_classification_model = None
_segmentation_model = None
_gradcam_model = None


def get_classification_model():
    """Lazy load classification model."""
    global _classification_model
    if _classification_model is None:
        print("Loading classification model...")
        # _classification_model = LFCBMInference(CLASSIFICATION_MODEL_PATH)
        _classification_model = ClassificationInference(RESNET_MODEL_PATH)

        print("Classification model loaded successfully!")
    return _classification_model


def get_segmentation_model():
    """Lazy load segmentation model."""
    global _segmentation_model
    if _segmentation_model is None:
        print("Loading ResUNet segmentation model...")
        _segmentation_model = ResUNetSegmentationInference(SEGMENTATION_MODEL_PATH)
        print("ResUNet segmentation model loaded successfully!")
    return _segmentation_model


def get_gradcam_model():
    """Lazy load Grad-CAM model."""
    global _gradcam_model
    if _gradcam_model is None:
        print("Loading Grad-CAM model...")
        # Use ResNet50 model for GradCAM (not LF-CBM)
        _gradcam_model = GradCAMInference(RESNET_MODEL_PATH)
        print("Grad-CAM model loaded successfully!")
    return _gradcam_model


def numpy_to_base64(img_array: np.ndarray) -> str:
    """Convert numpy array to base64 encoded string."""
    if img_array.dtype != np.uint8:
        img_array = (img_array * 255).astype(np.uint8)
    
    img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    
    return f"data:image/png;base64,{base64.b64encode(buffer.read()).decode('utf-8')}"


def decode_base64_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    image_bytes = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_bytes))


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'message': 'Brain MRI Analysis API is running',
        'models': {
            'classification': os.path.exists(CLASSIFICATION_MODEL_PATH),
            'segmentation': os.path.exists(SEGMENTATION_MODEL_PATH)
        }
    })


@app.route('/api/analyze', methods=['POST'])
def analyze_mri():
    """
    Main API endpoint for complete MRI analysis workflow.
    
    Accepts:
        - image: Base64 encoded image string OR file upload
        - include_gradcam (optional): Boolean to include Grad-CAM visualization (default: True)
        - include_segmentation (optional): Boolean to include segmentation (default: True)
        - segmentation_threshold (optional): Float threshold for segmentation mask (default: 0.5)
    
    Returns:
        JSON response with:
        - classification: {predicted_class, confidence, probabilities}
        - segmentation: {mask_base64, lesion_area_ratio, lesion_pixels}
        - gradcam: {overlay_base64}
        - success: Boolean
        - error: Error message if any
    """
    try:
        # Parse request data
        image = None
        include_gradcam = True
        include_segmentation = True
        segmentation_threshold = 0.5
        
        # Check for JSON payload
        if request.is_json:
            data = request.get_json()
            image_data = data.get('image')
            include_gradcam = data.get('include_gradcam', True)
            include_segmentation = data.get('include_segmentation', True)
            segmentation_threshold = data.get('segmentation_threshold', 0.5)
            
            if image_data:
                image = decode_base64_image(image_data)
        
        # Check for file upload
        elif 'image' in request.files:
            file = request.files['image']
            if file.filename != '':
                image = Image.open(file.stream).convert('RGB')
            
            # Get optional parameters from form data
            include_gradcam = request.form.get('include_gradcam', 'true').lower() == 'true'
            include_segmentation = request.form.get('include_segmentation', 'true').lower() == 'true'
            segmentation_threshold = float(request.form.get('segmentation_threshold', 0.5))
        
        if image is None:
            return jsonify({
                'success': False,
                'error': 'No image provided. Send image as base64 in JSON body or as file upload.'
            }), 400
        
        # Convert PIL Image to numpy array for processing
        image_np = np.array(image)
        
        response = {
            'success': True,
            'classification': None,
            'segmentation': None,
            'gradcam': None
        }
        
        # 1. Classification
        classification_model = get_classification_model()
        classification_result = classification_model.predict(image_np)
        response['classification'] = {
            'predicted_class': classification_result['predicted_class'],
            'predicted_index': classification_result['predicted_index'],
            'confidence': round(classification_result['confidence'] * 100, 2),  # Percentage
            'probabilities': {
                k: round(v * 100, 2) for k, v in classification_result['probabilities'].items()
            }
        }
        
        # 2. Segmentation (if requested)
        if include_segmentation:
            segmentation_model = get_segmentation_model()
            segmentation_result = segmentation_model.predict(
                image_np, 
                threshold=segmentation_threshold,
                return_overlay=True
            )
            response['segmentation'] = {
                'mask_base64': numpy_to_base64(segmentation_result['mask'] * 255),  # Convert binary to visible
                'overlay_base64': numpy_to_base64(segmentation_result['overlay']),
                'lesion_area_ratio': round(segmentation_result['lesion_area_ratio'] * 100, 4),  # Percentage
                'lesion_pixels': segmentation_result['lesion_pixels'],
                'total_pixels': segmentation_result['total_pixels']
            }
        
        # 3. Grad-CAM (if requested)
        if include_gradcam:
            gradcam_model = get_gradcam_model()
            gradcam_result = gradcam_model.generate_heatmap_overlay(image_np)
            response['gradcam'] = {
                'overlay_base64': numpy_to_base64(gradcam_result['overlay']),
                'target_class': gradcam_result['predicted_class']
            }
        
        return jsonify(response)
    
    except FileNotFoundError as e:
        return jsonify({
            'success': False,
            'error': f'Model file not found: {str(e)}'
        }), 500
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/classify', methods=['POST'])
def classify_only():
    """
    Classification-only endpoint (lightweight).
    
    Accepts:
        - image: Base64 encoded image string OR file upload
    
    Returns:
        JSON with classification results only.
    """
    try:
        image = None
        
        if request.is_json:
            data = request.get_json()
            image_data = data.get('image')
            if image_data:
                image = decode_base64_image(image_data)
        
        elif 'image' in request.files:
            file = request.files['image']
            if file.filename != '':
                image = Image.open(file.stream).convert('RGB')
        
        if image is None:
            return jsonify({
                'success': False,
                'error': 'No image provided.'
            }), 400
        
        image_np = np.array(image)
        classification_model = get_classification_model()
        result = classification_model.predict(image_np)
        
        return jsonify({
            'success': True,
            'predicted_class': result['predicted_class'],
            'predicted_index': result['predicted_index'],
            'confidence': round(result['confidence'] * 100, 2),
            'probabilities': {
                k: round(v * 100, 2) for k, v in result['probabilities'].items()
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/segment', methods=['POST'])
def segment_only():
    """
    Segmentation-only endpoint.
    
    Accepts:
        - image: Base64 encoded image string OR file upload
        - threshold (optional): Float threshold for mask (default: 0.5)
    
    Returns:
        JSON with segmentation results only.
    """
    try:
        image = None
        threshold = 0.5
        
        if request.is_json:
            data = request.get_json()
            image_data = data.get('image')
            threshold = data.get('threshold', 0.5)
            if image_data:
                image = decode_base64_image(image_data)
        
        elif 'image' in request.files:
            file = request.files['image']
            if file.filename != '':
                image = Image.open(file.stream).convert('RGB')
            threshold = float(request.form.get('threshold', 0.5))
        
        if image is None:
            return jsonify({
                'success': False,
                'error': 'No image provided.'
            }), 400
        
        image_np = np.array(image)
        segmentation_model = get_segmentation_model()
        result = segmentation_model.predict(image_np, threshold=threshold, return_overlay=True)
        
        return jsonify({
            'success': True,
            'mask_base64': numpy_to_base64(result['mask'] * 255),
            'overlay_base64': numpy_to_base64(result['overlay']),
            'lesion_area_ratio': round(result['lesion_area_ratio'] * 100, 4),
            'lesion_pixels': result['lesion_pixels'],
            'total_pixels': result['total_pixels']
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/gradcam', methods=['POST'])
def gradcam_only():
    """
    Grad-CAM visualization endpoint.
    
    Accepts:
        - image: Base64 encoded image string OR file upload
    
    Returns:
        JSON with Grad-CAM overlay and classification.
    """
    try:
        image = None
        
        if request.is_json:
            data = request.get_json()
            image_data = data.get('image')
            if image_data:
                image = decode_base64_image(image_data)
        
        elif 'image' in request.files:
            file = request.files['image']
            if file.filename != '':
                image = Image.open(file.stream).convert('RGB')
        
        if image is None:
            return jsonify({
                'success': False,
                'error': 'No image provided.'
            }), 400
        
        image_np = np.array(image)
        gradcam_model = get_gradcam_model()
        result = gradcam_model.generate_heatmap_overlay(image_np)
        
        return jsonify({
            'success': True,
            'overlay_base64': numpy_to_base64(result['overlay']),
            'predicted_class': result['predicted_class'],
            'predicted_index': result['predicted_index'],
            'confidence': round(result['confidence'] * 100, 2),
            'probabilities': {
                k: round(v * 100, 2) for k, v in result['probabilities'].items()
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("=" * 60)
    print("Brain MRI Classification & Segmentation API Server")
    print("=" * 60)
    print(f"Classification Model (LF-CBM): {CLASSIFICATION_MODEL_PATH}")
    print(f"ResNet Model (for GradCAM): {RESNET_MODEL_PATH}")
    print(f"Segmentation Model (ResUNet): {SEGMENTATION_MODEL_PATH}")
    print("=" * 60)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
