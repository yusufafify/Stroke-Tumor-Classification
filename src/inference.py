import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from typing import Dict, Tuple, Optional, Union, List
import warnings
from preprocess import preprocess_mri_image
from utils import decode_base64_image
import base64
import io
import open_clip


warnings.filterwarnings('ignore')




class LFCBMInference:
    """
    Inference class for Label-Free Concept Bottleneck Model.
    
    This class loads a pre-trained LF-CBM model and provides methods for
    making predictions with concept-based explanations.
    
    Attributes:
        device (torch.device): Device to run inference on (CPU or CUDA)
        model: BioMedCLIP vision-language model
        preprocess: Image preprocessing transforms
        tokenizer: Text tokenizer for concepts
        classifier: Trained logistic regression classifier
        class_names (List[str]): List of class names
        all_concepts (List[str]): List of all clinical concepts
        text_embeddings (torch.Tensor): Pre-computed text embeddings for concepts
    """
    
    def __init__(
        self, 
        model_path: str,
        device: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize the LF-CBM inference model.
        
        Args:
            model_path (str): Path to the saved .pth model file
            device (str, optional): Device to use ('cuda', 'cpu', or None for auto)
            verbose (bool): Whether to print loading information
        """
        self.verbose = verbose
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        if self.verbose:
            print(f"Initializing LF-CBM Inference on {self.device}")
        
        # Load the saved model
        self._load_model(model_path)
        
        # Load BioMedCLIP
        self._load_biomedclip()
        
        # Compute text embeddings
        self._compute_text_embeddings()
        
        if self.verbose:
            print("✓ LF-CBM Inference ready!")
    
    def _load_model(self, model_path: str):
        """Load the saved LF-CBM model and metadata."""
        if self.verbose:
            print(f"Loading model from {model_path}...")
        
        # Load model state
        model_state = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Extract components
        self.classifier = model_state['classifier']
        self.class_names = model_state['class_names']
        self.all_concepts = model_state['all_concepts']
        self.concept_bank = model_state['concept_bank']
        self.biomedclip_model_name = model_state['model_name']
        
        if self.verbose:
            print(f"  Classes: {self.class_names}")
            print(f"  Concepts: {len(self.all_concepts)}")
            print(f"  Test Accuracy: {model_state.get('test_accuracy', 'N/A')}")
    
    def _load_biomedclip(self):
        """Load BioMedCLIP model."""
        if self.verbose:
            print("Loading BioMedCLIP model...")
        
        # Load model using open_clip
        self.model, preprocess_train, self.preprocess = open_clip.create_model_and_transforms(
            self.biomedclip_model_name
        )
        self.tokenizer = open_clip.get_tokenizer(self.biomedclip_model_name)
        
        # Move to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        if self.verbose:
            print("✓ BioMedCLIP loaded")
    
    def _compute_text_embeddings(self):
        """Pre-compute text embeddings for all concepts."""
        if self.verbose:
            print("Computing text embeddings...")
        
        # Tokenize all concepts
        text_inputs = self.tokenizer(self.all_concepts).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            self.text_embeddings = self.model.encode_text(text_inputs)
            self.text_embeddings = F.normalize(self.text_embeddings, dim=-1)
        
        if self.verbose:
            print(f"✓ Text embeddings computed: {self.text_embeddings.shape}")
    
    def _load_image_from_base64(self, base64_string: str) -> Image.Image:
        """
        Load an image from a base64 encoded string.
        
        Args:
            base64_string (str): Base64 encoded image string
            
        Returns:
            PIL.Image.Image: Loaded image
        """
        # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
        if ',' in base64_string:
            base64_string = base64_string.split(',', 1)[1]
        
        # Decode base64 string
        image_data = base64.b64decode(base64_string)
        
        # Load image from bytes
        image = Image.open(io.BytesIO(image_data))
        
        return image
    
    def _load_image(self, image_input: Union[str, Image.Image, np.ndarray]) -> Image.Image:
        """
        Load an image from various input types.
        
        Args:
            image_input (Union[str, Image.Image, np.ndarray]): Can be:
                - File path (str)
                - Base64 encoded string (str starting with 'data:' or containing base64 data)
                - PIL Image object
                - Numpy array
            
        Returns:
            PIL.Image.Image: Loaded image
        """
        if isinstance(image_input, Image.Image):
            return image_input
        elif isinstance(image_input, np.ndarray):
            # Convert numpy array to PIL Image
            # Handle different array shapes and types
            if image_input.dtype != np.uint8:
                # Normalize to 0-255 range if needed
                if image_input.max() <= 1.0:
                    image_input = (image_input * 255).astype(np.uint8)
                else:
                    image_input = image_input.astype(np.uint8)
            return Image.fromarray(image_input)
        elif isinstance(image_input, str):
            # Check if it's a base64 string
            if image_input.startswith('data:') or (len(image_input) > 100 and not Path(image_input).exists()):
                return self._load_image_from_base64(image_input)
            else:
                # It's a file path
                return Image.open(image_input)
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
    
    def extract_concept_features(self, image_input: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Extract concept-based features from an image.
        
        Args:
            image_input (Union[str, Image.Image, np.ndarray]): Can be:
                - File path (str)
                - Base64 encoded string (str)
                - PIL Image object
                - Numpy array
            
        Returns:
            np.ndarray: Array of concept similarity scores
        """
        # Load and preprocess image
        image = self._load_image(image_input).convert("RGB")
        image_input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Get image embedding
        with torch.no_grad():
            image_embedding = self.model.encode_image(image_input_tensor)
            image_embedding = F.normalize(image_embedding, dim=-1)
        
        # Compute similarity with all concepts
        similarities = (image_embedding @ self.text_embeddings.T).squeeze(0)
        
        return similarities.cpu().numpy()
    
    def predict(
        self, 
        image_input: Union[str, Image.Image, np.ndarray],
        return_probabilities: bool = True,
        return_features: bool = False
    ) -> Dict:
        """
        Make a prediction for an image.
        
        Args:
            image_input (Union[str, Image.Image, np.ndarray]): Can be:
                - File path (str)
                - Base64 encoded string (str)
                - PIL Image object
                - Numpy array
            return_probabilities (bool): Whether to return class probabilities
            return_features (bool): Whether to return concept features
            
        Returns:
            Dict: Dictionary containing:
                - 'predicted_class' (str): Predicted class name
                - 'predicted_index' (int): Predicted class index (for app.py compatibility)
                - 'probabilities' (Dict): Class probabilities (if requested)
                - 'confidence' (float): Confidence score (max probability)
                - 'features' (np.ndarray): Concept features (if requested)
        """
        # Extract concept features
        features = self.extract_concept_features(image_input)
        
        # Make prediction
        prediction_idx = self.classifier.predict(features.reshape(1, -1))[0]
        predicted_class = self.class_names[prediction_idx]
        
        # Get probabilities
        probabilities = self.classifier.predict_proba(features.reshape(1, -1))[0]
        confidence = float(probabilities[prediction_idx])
        
        # Prepare result (using 'predicted_index' to match app.py expectations)
        result = {
            'predicted_class': predicted_class,
            'predicted_index': int(prediction_idx),  # Changed from predicted_idx for app.py compatibility
            'confidence': confidence
        }
        
        if return_probabilities:
            result['probabilities'] = {
                class_name: float(prob) 
                for class_name, prob in zip(self.class_names, probabilities)
            }
        
        if return_features:
            result['features'] = features
        
        return result
    
    def explain_prediction(
        self, 
        image_input: Union[str, Image.Image, np.ndarray],
        top_k: int = 5
    ) -> Dict:
        """
        Make a prediction with detailed concept-based explanation.
        
        Args:
            image_input (Union[str, Image.Image, np.ndarray]): Can be:
                - File path (str)
                - Base64 encoded string (str)
                - PIL Image object
                - Numpy array
            top_k (int): Number of top concepts to include in explanation
            
        Returns:
            Dict: Dictionary containing:
                - 'predicted_class' (str): Predicted class name
                - 'confidence' (float): Prediction confidence
                - 'probabilities' (Dict): All class probabilities
                - 'top_concepts' (List[Dict]): Top activated concepts with details
                - 'top_supporting_concepts' (List[Dict]): Top concepts supporting the prediction
                - 'interpretation' (str): Human-readable explanation
        """
        # Get basic prediction
        result = self.predict(image_input, return_probabilities=True, return_features=True)
        features = result.pop('features')
        predicted_idx = result['predicted_idx']
        predicted_class = result['predicted_class']
        
        # Get top activated concepts
        concept_scores = list(zip(self.all_concepts, features))
        concept_scores.sort(key=lambda x: x[1], reverse=True)
        
        top_concepts = []
        for i, (concept, score) in enumerate(concept_scores[:top_k]):
            concept_idx = self.all_concepts.index(concept)
            weight = self.classifier.coef_[predicted_idx, concept_idx]
            contribution = score * weight
            
            top_concepts.append({
                'concept': concept,
                'similarity_score': float(score),
                'weight': float(weight),
                'contribution': float(contribution)
            })
        
        # Get top supporting concepts (high score AND positive weight)
        contributions = []
        for i, concept in enumerate(self.all_concepts):
            score = features[i]
            weight = self.classifier.coef_[predicted_idx, i]
            contrib = score * weight
            contributions.append({
                'concept': concept,
                'similarity_score': float(score),
                'weight': float(weight),
                'contribution': float(contrib)
            })
        
        contributions.sort(key=lambda x: x['contribution'], reverse=True)
        top_supporting = [c for c in contributions[:top_k] if c['contribution'] > 0]
        
        # Generate interpretation
        interpretation = f"The model predicted '{predicted_class}' with {result['confidence']:.2%} confidence. "
        
        if top_supporting:
            interpretation += "Key supporting evidence includes: "
            evidence_parts = []
            for i, c in enumerate(top_supporting[:3], 1):
                evidence_parts.append(
                    f"{c['concept']} (similarity: {c['similarity_score']:.3f})"
                )
            interpretation += ", ".join(evidence_parts) + "."
        
        # Build final result
        result['top_concepts'] = top_concepts
        result['top_supporting_concepts'] = top_supporting
        result['interpretation'] = interpretation
        
        return result
    
    def predict_batch(
        self, 
        image_paths: List[str],
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Make predictions for multiple images.
        
        Args:
            image_paths (List[str]): List of image file paths
            show_progress (bool): Whether to show progress bar
            
        Returns:
            List[Dict]: List of prediction results for each image
        """
        results = []
        
        iterator = image_paths
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(image_paths, desc="Processing images")
            except ImportError:
                pass
        
        for image_path in iterator:
            try:
                result = self.predict(image_path, return_probabilities=True)
                result['image_path'] = image_path
                result['status'] = 'success'
            except Exception as e:
                result = {
                    'image_path': image_path,
                    'status': 'error',
                    'error': str(e)
                }
            results.append(result)
        
        return results
    
    def get_concept_importance(self) -> Dict[str, np.ndarray]:
        """
        Get concept importance weights for each class.
        
        Returns:
            Dict[str, np.ndarray]: Dictionary mapping class names to concept weights
        """
        importance = {}
        for i, class_name in enumerate(self.class_names):
            importance[class_name] = self.classifier.coef_[i]
        return importance
    
    def get_top_concepts_for_class(
        self, 
        class_name: str, 
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get the most important concepts for a specific class.
        
        Args:
            class_name (str): Name of the class
            top_k (int): Number of top concepts to return
            
        Returns:
            List[Tuple[str, float]]: List of (concept, weight) tuples
        """
        if class_name not in self.class_names:
            raise ValueError(f"Unknown class: {class_name}")
        
        class_idx = self.class_names.index(class_name)
        weights = self.classifier.coef_[class_idx]
        
        # Get top by absolute value
        top_indices = np.argsort(np.abs(weights))[-top_k:][::-1]
        
        return [(self.all_concepts[i], weights[i]) for i in top_indices]
    
    def __repr__(self):
        return (
            f"LFCBMInference(\n"
            f"  device={self.device},\n"
            f"  classes={self.class_names},\n"
            f"  n_concepts={len(self.all_concepts)}\n"
            f")"
        )


class ClassificationInference:
    """
    Inference class for brain scan classification (Hemorrhagic, Ischemic, Tumor)
    Uses ResNet50 architecture.
    """
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize the classification model.
        
        Args:
            model_path: Path to the trained classification model (.pth file)
            device: Device to run inference on ('cuda', 'cpu', or None for auto-detect)
        """
        self.classes = ["Hemorrhagic", "Ischemic", "Tumor"]
        self.num_classes = len(self.classes)
        self.device = self._get_device(device)
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()
        
    def _get_device(self, device: Optional[str]) -> torch.device:
        """Determine the device to use for inference."""
        if device:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load the trained ResNet50 model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Initialize ResNet50 architecture
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.num_classes)
        
        # Load trained weights
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _get_transform(self) -> transforms.Compose:
        """Get the image transformation pipeline."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Preprocess image for model input with MRI-specific preprocessing.
        Pipeline: Load -> ROI Extraction -> CLAHE -> Resize -> ToTensor -> Normalize
        
        Args:
            image: Input image (file path, PIL Image, numpy array, or base64 string)
            
        Returns:
            Preprocessed image tensor
        """
        # Load image and convert to OpenCV format (BGR) for preprocessing
        if isinstance(image, str):
            # Check if it's a base64 string
            if image.startswith('data:image') or len(image) > 500:  # Heuristic for base64
                try:
                    pil_image = decode_base64_image(image)
                    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    raise ValueError(f"Failed to decode base64 image: {e}")
            elif not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            else:
                # Load as OpenCV image (BGR)
                cv_image = cv2.imread(image)
                if cv_image is None:
                    raise ValueError(f"Failed to load image: {image}")
        
        elif isinstance(image, Image.Image):
            # Convert PIL to OpenCV (BGR)
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        elif isinstance(image, np.ndarray):
            # Assume it's already in BGR format (OpenCV standard)
            # If it's RGB, you might need to convert: cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv_image = image if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        # Apply MRI preprocessing: ROI extraction + CLAHE normalization
        preprocessed_cv = preprocess_mri_image(cv_image)
        
        # Convert back to PIL Image for PyTorch transforms
        preprocessed_rgb = cv2.cvtColor(preprocessed_cv, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(preprocessed_rgb)
        
        # Apply PyTorch transformations (Resize, ToTensor, Normalize)
        image_tensor = self.transform(pil_image)
        return image_tensor.unsqueeze(0)  # Add batch dimension
    
    def predict(self, image: Union[str, Image.Image, np.ndarray]) -> Dict:
        """
        Perform classification on a single image.
        
        Args:
            image: Input image (file path, PIL Image, or numpy array)
            
        Returns:
            Dictionary containing:
                - predicted_class: Name of predicted class
                - predicted_index: Index of predicted class
                - confidence: Confidence score (0-1)
                - probabilities: Dictionary of all class probabilities
        """
        # Preprocess image
        input_tensor = self.preprocess_image(image).to(self.device)
        
        # Perform inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, 0)
        
        # Prepare results
        result = {
            'predicted_class': self.classes[predicted_idx.item()],
            'predicted_index': predicted_idx.item(),
            'confidence': confidence.item(),
            'probabilities': {
                self.classes[i]: prob.item() 
                for i, prob in enumerate(probabilities)
            }
        }
        
        return result


class GradCAMInference:
    """
    Standalone Grad-CAM inference class for generating heatmap visualizations.
    Takes an image, preprocesses it, generates Grad-CAM heatmap, and returns image with overlay.
    """
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize Grad-CAM inference.
        
        Args:
            model_path: Path to the trained classification model (.pth file)
            device: Device to run inference on ('cuda', 'cpu', or None for auto-detect)
        """
        self.classes = ["Hemorrhagic", "Ischemic", "Tumor"]
        self.num_classes = len(self.classes)
        self.device = self._get_device(device)
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()
        
        # Setup Grad-CAM hooks
        self.target_layer = self.model.layer4[-1]  # Last conv layer in ResNet50
        self.gradients = None
        self.activations = None
        self.forward_hook = self.target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = self.target_layer.register_full_backward_hook(self._save_gradient)
    
    def _get_device(self, device: Optional[str]) -> torch.device:
        """Determine the device to use for inference."""
        if device:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load the trained ResNet50 model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.num_classes)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model
    
    def _get_transform(self) -> transforms.Compose:
        """Get the image transformation pipeline."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _save_activation(self, module, input, output):
        """Hook to save activations during forward pass."""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save gradients during backward pass."""
        self.gradients = grad_output[0].detach()
    
    def preprocess_image(self, image: Union[str, Image.Image, np.ndarray]) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Preprocess image for model input with MRI-specific preprocessing.
        Pipeline: Load -> ROI Extraction -> CLAHE -> Resize -> ToTensor -> Normalize
        
        Args:
            image: Input image (file path, PIL Image, numpy array, or base64 string)
            
        Returns:
            Tuple of (preprocessed tensor, original image array in RGB)
        """
        # Load and convert to OpenCV format
        if isinstance(image, str):
            # Check if it's a base64 string
            if image.startswith('data:image') or len(image) > 500:  # Heuristic for base64
                try:
                    pil_image = decode_base64_image(image)
                    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    raise ValueError(f"Failed to decode base64 image: {e}")
            elif not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            else:
                cv_image = cv2.imread(image)
                if cv_image is None:
                    raise ValueError(f"Failed to load image: {image}")
        elif isinstance(image, Image.Image):
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            cv_image = image if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        # Apply MRI preprocessing
        preprocessed_cv = preprocess_mri_image(cv_image)
        preprocessed_rgb = cv2.cvtColor(preprocessed_cv, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL for transforms
        pil_image = Image.fromarray(preprocessed_rgb)
        image_tensor = self.transform(pil_image)
        
        return image_tensor.unsqueeze(0), preprocessed_rgb
    
    def generate_heatmap_overlay(
        self, 
        image: Union[str, Image.Image, np.ndarray],
        target_class: Optional[int] = None,
        alpha: float = 0.4
    ) -> Dict:
        """
        Generate Grad-CAM heatmap and overlay it on the original image.
        
        Args:
            image: Input image (file path, PIL Image, or numpy array)
            target_class: Target class index for Grad-CAM (if None, uses predicted class)
            alpha: Transparency of heatmap overlay (0-1)
            
        Returns:
            Dictionary containing:
                - predicted_class: Name of predicted class
                - predicted_index: Index of predicted class
                - confidence: Confidence score (0-1)
                - probabilities: Dictionary of all class probabilities
                - heatmap: Raw Grad-CAM heatmap (normalized 0-1)
                - overlay: Image with heatmap overlay (RGB numpy array)
        """
        # Preprocess image
        input_tensor, original_array = self.preprocess_image(image)
        input_tensor = input_tensor.to(self.device)
        
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        probabilities = F.softmax(output, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)
        
        # Use predicted class if target not specified
        if target_class is None:
            target_class = predicted_idx.item()
        
        # Backward pass for Grad-CAM
        self.model.zero_grad()
        target = output[0, target_class]
        target.backward()
        
        # Calculate Grad-CAM
        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam_np = cam.cpu().numpy()
        
        # Resize CAM to match original image
        cam_resized = cv2.resize(cam_np, (original_array.shape[1], original_array.shape[0]))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Create overlay
        overlay = (heatmap * alpha + original_array * (1 - alpha)).astype(np.uint8)
        
        return {
            'predicted_class': self.classes[predicted_idx.item()],
            'predicted_index': predicted_idx.item(),
            'confidence': confidence.item(),
            'probabilities': {
                self.classes[i]: prob.item() 
                for i, prob in enumerate(probabilities)
            },
            'heatmap': cam_resized,
            'overlay': overlay
        }
    
    def __del__(self):
        """Cleanup hooks."""
        try:
            self.forward_hook.remove()
            self.backward_hook.remove()
        except:
            pass


class UNet(nn.Module):
    """
    U-Net architecture for brain lesion segmentation.
    """
    
    def __init__(self):
        super(UNet, self).__init__()
        
        # Helper to create double convolutions
        def double_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        # Encoder (Downsampling)
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)

        # Decoder (Upsampling)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(64 + 128, 64)
        
        self.conv_last = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Down
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        
        x = self.dconv_down4(x)
        
        # Up
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        return torch.sigmoid(out)


class SegmentationInference:
    """
    Inference class for brain lesion segmentation using U-Net.
    """
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize the segmentation model.
        
        Args:
            model_path: Path to the trained segmentation model (.pth file)
            device: Device to run inference on ('cuda', 'cpu', or None for auto-detect)
        """
        self.device = self._get_device(device)
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()
        
    def _get_device(self, device: Optional[str]) -> torch.device:
        """Determine the device to use for inference."""
        if device:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load the trained U-Net model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Initialize U-Net architecture
        model = UNet()
        
        # Load trained weights
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _get_transform(self) -> transforms.Compose:
        """Get the image transformation pipeline."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image: Union[str, Image.Image, np.ndarray]) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Preprocess image for model input with MRI-specific preprocessing.
        Pipeline: Load -> ROI Extraction -> CLAHE -> Resize -> ToTensor -> Normalize
        
        Args:
            image: Input image (file path, PIL Image, numpy array, or base64 string)
            
        Returns:
            Tuple of (preprocessed tensor, original preprocessed image array in RGB)
        """
        # Load image and convert to OpenCV format (BGR) for preprocessing
        if isinstance(image, str):
            # Check if it's a base64 string
            if image.startswith('data:image') or len(image) > 500:  # Heuristic for base64
                try:
                    pil_image = decode_base64_image(image)
                    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    raise ValueError(f"Failed to decode base64 image: {e}")
            elif not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            else:
                # Load as OpenCV image (BGR)
                cv_image = cv2.imread(image)
                if cv_image is None:
                    raise ValueError(f"Failed to load image: {image}")
        
        elif isinstance(image, Image.Image):
            # Convert PIL to OpenCV (BGR)
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        elif isinstance(image, np.ndarray):
            # Assume it's already in BGR format (OpenCV standard)
            cv_image = image if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        # Apply MRI preprocessing: ROI extraction + CLAHE normalization
        preprocessed_cv = preprocess_mri_image(cv_image)
        
        # Convert to RGB for consistency
        preprocessed_rgb = cv2.cvtColor(preprocessed_cv, cv2.COLOR_BGR2RGB)
        original_array = preprocessed_rgb.copy()  # Keep the preprocessed version
        
        # Convert to PIL for PyTorch transforms
        pil_image = Image.fromarray(preprocessed_rgb)
        
        # Apply PyTorch transformations (Resize, ToTensor, Normalize)
        image_tensor = self.transform(pil_image)
        return image_tensor.unsqueeze(0), original_array
    
    def predict(self, image: Union[str, Image.Image, np.ndarray], 
                threshold: float = 0.5, 
                return_overlay: bool = False) -> Dict:
        """
        Perform segmentation on a single image.
        
        Args:
            image: Input image (file path, PIL Image, or numpy array)
            threshold: Threshold for binary mask (0-1)
            return_overlay: Whether to return overlay visualization
            
        Returns:
            Dictionary containing:
                - mask: Binary segmentation mask (numpy array)
                - probability_map: Raw probability map (numpy array)
                - lesion_area_ratio: Ratio of lesion area to total brain area
                - overlay: (Optional) RGB overlay visualization
        """
        # Preprocess image
        input_tensor, original_array = self.preprocess_image(image)
        input_tensor = input_tensor.to(self.device)
        
        # Perform inference
        with torch.no_grad():
            output = self.model(input_tensor)
            probability_map = output.cpu().squeeze().numpy()
        
        # Create binary mask
        binary_mask = (probability_map > threshold).astype(np.uint8)
        
        # Resize masks to original image size
        original_size = (original_array.shape[1], original_array.shape[0])  # (width, height)
        probability_map_resized = cv2.resize(probability_map, original_size)
        binary_mask_resized = cv2.resize(binary_mask, original_size, interpolation=cv2.INTER_NEAREST)
        
        # Calculate lesion area ratio
        total_pixels = binary_mask_resized.size
        lesion_pixels = np.sum(binary_mask_resized)
        lesion_area_ratio = lesion_pixels / total_pixels
        
        # Prepare result
        result = {
            'mask': binary_mask_resized,
            'probability_map': probability_map_resized,
            'lesion_area_ratio': float(lesion_area_ratio),
            'lesion_pixels': int(lesion_pixels),
            'total_pixels': int(total_pixels)
        }
        
        # Create overlay if requested
        if return_overlay:
            overlay = self._create_overlay(original_array, binary_mask_resized)
            result['overlay'] = overlay
        
        return result
    
    def _create_overlay(self, original_image: np.ndarray, mask: np.ndarray, 
                        color: Tuple[int, int, int] = (255, 0, 0), 
                        alpha: float = 0.4) -> np.ndarray:
        """
        Create an overlay visualization of the mask on the original image.
        
        Args:
            original_image: Original RGB image
            mask: Binary mask
            color: RGB color for the mask overlay
            alpha: Transparency of the overlay (0-1)
            
        Returns:
            Overlay image as numpy array
        """
        # Create colored mask
        colored_mask = np.zeros_like(original_image)
        colored_mask[mask > 0] = color
        
        # Blend with original image
        overlay = cv2.addWeighted(original_image, 1 - alpha, colored_mask, alpha, 0)
        
        return overlay


# ============================================================================
# ResUNet Architecture for Advanced Segmentation
# ============================================================================

class ResidualBlock(nn.Module):
    """Residual block with skip connection for ResUNet"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection - adjust dimensions if needed
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity  # Residual connection
        out = self.relu(out)
        
        return out


class ResUNet(nn.Module):
    """Residual U-Net for medical image segmentation"""
    def __init__(self, in_channels=3, out_channels=1):
        super(ResUNet, self).__init__()
        
        # Initial convolution
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Encoder (Downsampling path)
        self.encoder1 = ResidualBlock(64, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.encoder2 = ResidualBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.encoder3 = ResidualBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.encoder4 = ResidualBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bridge (Bottleneck)
        self.bridge = ResidualBlock(512, 1024)
        
        # Decoder (Upsampling path)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = ResidualBlock(1024, 512)  # 1024 = 512 (upconv) + 512 (skip)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = ResidualBlock(512, 256)  # 512 = 256 + 256
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = ResidualBlock(256, 128)  # 256 = 128 + 128
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = ResidualBlock(128, 64)  # 128 = 64 + 64
        
        # Output layer
        self.output_layer = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x = self.input_layer(x)
        
        enc1 = self.encoder1(x)
        x = self.pool1(enc1)
        
        enc2 = self.encoder2(x)
        x = self.pool2(enc2)
        
        enc3 = self.encoder3(x)
        x = self.pool3(enc3)
        
        enc4 = self.encoder4(x)
        x = self.pool4(enc4)
        
        # Bridge
        x = self.bridge(x)
        
        # Decoder with skip connections
        x = self.upconv4(x)
        x = torch.cat([x, enc4], dim=1)  # Skip connection
        x = self.decoder4(x)
        
        x = self.upconv3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.decoder3(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.decoder2(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.decoder1(x)
        
        # Output
        x = self.output_layer(x)
        return torch.sigmoid(x)


class ResUNetSegmentationInference:
    """
    Inference class for brain lesion segmentation using ResUNet.
    Provides advanced segmentation with residual connections for better feature extraction.
    """
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize the ResUNet segmentation model.
        
        Args:
            model_path: Path to the trained ResUNet model (.pth file)
            device: Device to run inference on ('cuda', 'cpu', or None for auto-detect)
        """
        self.device = self._get_device(device)
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()
        
    def _get_device(self, device: Optional[str]) -> torch.device:
        """Determine the device to use for inference."""
        if device:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load the trained ResUNet model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Initialize ResUNet architecture
        model = ResUNet()
        
        # Load trained weights
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _get_transform(self) -> transforms.Compose:
        """Get the image transformation pipeline."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image: Union[str, Image.Image, np.ndarray]) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Preprocess image for model input with MRI-specific preprocessing.
        Pipeline: Load -> ROI Extraction -> CLAHE -> Resize -> ToTensor -> Normalize
        
        Args:
            image: Input image (file path, PIL Image, numpy array, or base64 string)
            
        Returns:
            Tuple of (preprocessed tensor, original preprocessed image array in RGB)
        """
        # Load image and convert to OpenCV format (BGR) for preprocessing
        if isinstance(image, str):
            # Check if it's a base64 string
            if image.startswith('data:image') or len(image) > 500:
                try:
                    pil_image = decode_base64_image(image)
                    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    raise ValueError(f"Failed to decode base64 image: {e}")
            elif not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            else:
                cv_image = cv2.imread(image)
                if cv_image is None:
                    raise ValueError(f"Failed to load image: {image}")
        
        elif isinstance(image, Image.Image):
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        elif isinstance(image, np.ndarray):
            cv_image = image if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        # Apply MRI preprocessing: ROI extraction + CLAHE normalization
        preprocessed_cv = preprocess_mri_image(cv_image)
        
        # Convert to RGB for consistency
        preprocessed_rgb = cv2.cvtColor(preprocessed_cv, cv2.COLOR_BGR2RGB)
        original_array = preprocessed_rgb.copy()
        
        # Convert to PIL for PyTorch transforms
        pil_image = Image.fromarray(preprocessed_rgb)
        
        # Apply PyTorch transformations
        image_tensor = self.transform(pil_image)
        return image_tensor.unsqueeze(0), original_array
    
    def predict(self, image: Union[str, Image.Image, np.ndarray], 
                threshold: float = 0.5, 
                return_overlay: bool = False) -> Dict:
        """
        Perform segmentation on a single image using ResUNet.
        
        Args:
            image: Input image (file path, PIL Image, or numpy array)
            threshold: Threshold for binary mask (0-1)
            return_overlay: Whether to return overlay visualization
            
        Returns:
            Dictionary containing:
                - mask: Binary segmentation mask (numpy array)
                - probability_map: Raw probability map (numpy array)
                - lesion_area_ratio: Ratio of lesion area to total brain area
                - overlay: (Optional) RGB overlay visualization
        """
        # Preprocess image
        input_tensor, original_array = self.preprocess_image(image)
        input_tensor = input_tensor.to(self.device)
        
        # Perform inference
        with torch.no_grad():
            output = self.model(input_tensor)
            probability_map = output.cpu().squeeze().numpy()
        
        # Create binary mask
        binary_mask = (probability_map > threshold).astype(np.uint8)
        
        # Resize masks to original image size
        original_size = (original_array.shape[1], original_array.shape[0])
        probability_map_resized = cv2.resize(probability_map, original_size)
        binary_mask_resized = cv2.resize(binary_mask, original_size, interpolation=cv2.INTER_NEAREST)
        
        # Calculate lesion area ratio
        total_pixels = binary_mask_resized.size
        lesion_pixels = np.sum(binary_mask_resized)
        lesion_area_ratio = lesion_pixels / total_pixels
        
        # Prepare result
        result = {
            'mask': binary_mask_resized,
            'probability_map': probability_map_resized,
            'lesion_area_ratio': float(lesion_area_ratio),
            'lesion_pixels': int(lesion_pixels),
            'total_pixels': int(total_pixels)
        }
        
        # Create overlay if requested
        if return_overlay:
            overlay = self._create_overlay(original_array, binary_mask_resized)
            result['overlay'] = overlay
        
        return result
    
    def _create_overlay(self, original_image: np.ndarray, mask: np.ndarray, 
                        color: Tuple[int, int, int] = (255, 0, 0), 
                        alpha: float = 0.4) -> np.ndarray:
        """
        Create an overlay visualization of the mask on the original image.
        
        Args:
            original_image: Original RGB image
            mask: Binary mask
            color: RGB color for the mask overlay
            alpha: Transparency of the overlay (0-1)
            
        Returns:
            Overlay image as numpy array
        """
        # Create colored mask
        colored_mask = np.zeros_like(original_image)
        colored_mask[mask > 0] = color
        
        # Blend with original image
        overlay = cv2.addWeighted(original_image, 1 - alpha, colored_mask, alpha, 0)
        
        return overlay