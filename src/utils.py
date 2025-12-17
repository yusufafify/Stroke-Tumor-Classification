import base64
import io
from PIL import Image

def decode_base64_image(base64_string: str) -> Image.Image:
    """
    Decode a base64 encoded image string to PIL Image.
    
    Args:
        base64_string: Base64 encoded image string (with or without data URI prefix)
        
    Returns:
        PIL Image object
    """
    # Remove data URI prefix if present (e.g., "data:image/jpeg;base64,")
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # Decode base64 to bytes
    image_bytes = base64.b64decode(base64_string)
    
    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    return image

