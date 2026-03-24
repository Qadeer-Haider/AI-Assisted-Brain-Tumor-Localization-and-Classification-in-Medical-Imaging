import numpy as np
import cv2
from PIL import Image

# Class definitions (update based on project details)
# Based on typical brain tumor datasets like in this project, assuming:
# For classification: e.g., glioma, meningioma, no tumor, pituitary
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

def preprocess_image(image, target_size=(224, 224), task='classification'):
    """Preprocess uploaded image for model inference."""
    # Convert PIL image to numpy array (RGB)
    img_np = np.array(image.convert('RGB'))
    
    # Resize image
    img_resized = cv2.resize(img_np, target_size)
    
    # Normalization
    img_normalized = img_resized / 255.0  # Common normalization for both models
    
    if task == 'segmentation':
        # For segmentation, input might be 256x256, adjust if necessary
        # Usually models take shapes like (1, H, W, 3)
        return np.expand_dims(img_normalized, axis=0).astype(np.float32)
    else:
        # Classification
        return np.expand_dims(img_normalized, axis=0).astype(np.float32)

def decode_classification(prediction):
    """Decode the outputs of the classification model."""
    class_idx = np.argmax(prediction[0])
    confidence = prediction[0][class_idx]
    
    return CLASS_NAMES[class_idx], confidence

def postprocess_segmentation(prediction, original_size, threshold=0.5):
    """Decode the outputs of the segmentation model and apply it as a mask."""
    # prediction shape usually (1, H, W, 1) or (1, H, W, num_classes)
    mask = prediction[0]
    
    if mask.shape[-1] == 1:
        mask = (mask > threshold).astype(np.uint8) * 255
    else:
        # Multiclass segmentation
        mask = np.argmax(mask, axis=-1).astype(np.uint8) * 255
        
    # Resize mask to original image size
    mask_resized = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
    return mask_resized

def overlay_mask(image, mask, alpha=0.5, color=(255, 0, 0)):
    """Overlay a binary mask onto the original PIL Image."""
    img_np = np.array(image.convert('RGB'))
    
    # Create colored mask
    colored_mask = np.zeros_like(img_np)
    colored_mask[mask > 0] = color
    
    # Overlay
    output_image = cv2.addWeighted(img_np, 1.0, colored_mask, alpha, 0)
    return Image.fromarray(output_image)
