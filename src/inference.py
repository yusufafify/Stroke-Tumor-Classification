import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from typing import Dict, Tuple, Optional, Union
import warnings
from preprocess import preprocess_mri_image
from utils import decode_base64_image

warnings.filterwarnings('ignore')


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