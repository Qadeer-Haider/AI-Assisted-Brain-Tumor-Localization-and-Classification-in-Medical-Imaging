"""
Tumor Segmentation Predictor.

Provides inference and visualization utilities for trained segmentation models.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf


class TumorSegmentor:
    """
    Inference class for brain tumor segmentation.
    
    Provides methods for loading trained models and predicting tumor masks
    with visualization utilities.
    
    Example:
        >>> segmentor = TumorSegmentor("weights/UNet_bce_tversky_best.keras")
        >>> mask = segmentor.predict("brain_scan.jpg")
        >>> segmentor.visualize("brain_scan.jpg", overlay=True)
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[tf.keras.Model] = None,
        img_size: Tuple[int, int] = (256, 256),
        threshold: float = 0.5,
    ):
        """
        Initialize the segmentor.
        
        Args:
            model_path: Path to a saved model file.
            model: Pre-loaded Keras model (alternative to model_path).
            img_size: Target image size for preprocessing.
            threshold: Threshold for binary mask conversion.
        """
        self.img_size = img_size
        self.threshold = threshold
        
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = self.load_model(model_path)
        else:
            self.model = None
    
    def load_model(self, model_path: str) -> tf.keras.Model:
        """
        Load a trained segmentation model.
        
        Args:
            model_path: Path to the saved model.
            
        Returns:
            Loaded Keras model.
        """
        # Import custom objects
        custom_objects = {}
        
        try:
            from ..training.metrics import (
                dice_coefficient, iou_score, sensitivity, specificity, precision_metric,
                dice_loss_metric, tversky_loss_metric, focal_tversky_loss_metric,
                dice_bce_loss_metric, bce_tversky_loss_metric,
            )
            custom_objects.update({
                "dice_coefficient": dice_coefficient,
                "iou_score": iou_score,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "precision_metric": precision_metric,
                "dice_loss_metric": dice_loss_metric,
                "tversky_loss_metric": tversky_loss_metric,
                "focal_tversky_loss_metric": focal_tversky_loss_metric,
                "dice_bce_loss_metric": dice_bce_loss_metric,
                "bce_tversky_loss_metric": bce_tversky_loss_metric,
            })
        except ImportError:
            pass
        
        try:
            from ..training.losses import LOSS_FUNCTIONS
            custom_objects.update(LOSS_FUNCTIONS)
        except ImportError:
            pass
        
        self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print(f"✅ Model loaded from: {model_path}")
        
        return self.model
    
    def preprocess_image(self, image: Union[str, np.ndarray]) -> tf.Tensor:
        """
        Preprocess an image for inference.
        
        Args:
            image: Image path or numpy array.
            
        Returns:
            Preprocessed image tensor with batch dimension.
        """
        if isinstance(image, str):
            # Load from file
            img = tf.io.read_file(image)
            img = tf.image.decode_png(img, channels=3)
        else:
            img = tf.constant(image, dtype=tf.float32)
        
        # Resize
        img = tf.image.resize(img, self.img_size)
        
        # Normalize to [0, 1]
        if tf.reduce_max(img) > 1.0:
            img = img / 255.0
        
        # Add batch dimension
        img = tf.expand_dims(img, axis=0)
        
        return img
    
    def predict(
        self,
        image: Union[str, np.ndarray],
        return_probabilities: bool = False,
    ) -> np.ndarray:
        """
        Predict tumor segmentation mask for an image.
        
        Args:
            image: Image path or numpy array.
            return_probabilities: If True, return probability map instead of binary mask.
            
        Returns:
            Predicted mask as numpy array (H, W, 1).
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        # Preprocess
        img = self.preprocess_image(image)
        
        # Predict
        pred = self.model.predict(img, verbose=0)
        
        # Remove batch dimension
        pred = pred[0]
        
        if not return_probabilities:
            # Apply threshold for binary mask
            pred = (pred > self.threshold).astype(np.float32)
        
        return pred
    
    def predict_batch(
        self,
        images: list,
        return_probabilities: bool = False,
    ) -> np.ndarray:
        """
        Predict tumor segmentation masks for a batch of images.
        
        Args:
            images: List of image paths or numpy arrays.
            return_probabilities: If True, return probability maps.
            
        Returns:
            Array of predicted masks (N, H, W, 1).
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        # Preprocess all images
        batch = []
        for img in images:
            preprocessed = self.preprocess_image(img)
            batch.append(preprocessed[0])
        
        batch = np.stack(batch, axis=0)
        
        # Predict
        preds = self.model.predict(batch, verbose=0)
        
        if not return_probabilities:
            preds = (preds > self.threshold).astype(np.float32)
        
        return preds
    
    def overlay_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        color: Tuple[int, int, int] = (255, 0, 0),
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Overlay a segmentation mask on an image.
        
        Args:
            image: Original image (H, W, 3).
            mask: Binary mask (H, W, 1) or (H, W).
            color: RGB color for the overlay.
            alpha: Transparency of the overlay.
            
        Returns:
            Image with mask overlay (H, W, 3).
        """
        # Ensure proper shapes
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        # Normalize image to 0-255 if needed
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # Create colored overlay
        overlay = image.copy()
        color_mask = np.zeros_like(image)
        color_mask[mask > 0.5] = color
        
        # Blend
        overlay = np.where(
            mask[:, :, np.newaxis] > 0.5,
            (alpha * color_mask + (1 - alpha) * image).astype(np.uint8),
            image,
        )
        
        return overlay
    
    def visualize(
        self,
        image: Union[str, np.ndarray],
        save_path: Optional[str] = None,
        overlay: bool = True,
        show: bool = True,
    ):
        """
        Visualize prediction on an image.
        
        Args:
            image: Image path or numpy array.
            save_path: Optional path to save the visualization.
            overlay: If True, overlay mask on image. If False, show side by side.
            show: If True, display the visualization.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠️ matplotlib not installed. Cannot visualize.")
            return
        
        # Load image if path
        if isinstance(image, str):
            img = tf.io.read_file(image)
            img = tf.image.decode_png(img, channels=3)
            img = tf.image.resize(img, self.img_size).numpy()
            img = img / 255.0 if img.max() > 1 else img
        else:
            img = image
            if img.max() > 1:
                img = img / 255.0
        
        # Predict
        mask = self.predict(image)
        prob = self.predict(image, return_probabilities=True)
        
        if overlay:
            # Single overlay visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(img)
            axes[0].set_title("Original")
            axes[0].axis("off")
            
            axes[1].imshow(prob[:, :, 0], cmap="hot")
            axes[1].set_title("Probability Map")
            axes[1].axis("off")
            
            overlay_img = self.overlay_mask(img, mask)
            axes[2].imshow(overlay_img)
            axes[2].set_title("Overlay")
            axes[2].axis("off")
        else:
            # Side by side
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            axes[0].imshow(img)
            axes[0].set_title("Original")
            axes[0].axis("off")
            
            axes[1].imshow(mask[:, :, 0], cmap="gray")
            axes[1].set_title("Binary Mask")
            axes[1].axis("off")
            
            axes[2].imshow(prob[:, :, 0], cmap="hot")
            axes[2].set_title("Probability Map")
            axes[2].axis("off")
            
            overlay_img = self.overlay_mask(img, mask)
            axes[3].imshow(overlay_img)
            axes[3].set_title("Overlay")
            axes[3].axis("off")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"💾 Visualization saved to: {save_path}")
        
        if show:
            plt.show()
        
        plt.close()


def predict_mask(
    image_path: str,
    model_path: str,
    output_path: Optional[str] = None,
    visualize: bool = False,
) -> np.ndarray:
    """
    Convenience function to predict tumor mask for a single image.
    
    Args:
        image_path: Path to the input image.
        model_path: Path to the trained model.
        output_path: Optional path to save the mask.
        visualize: If True, show visualization.
        
    Returns:
        Predicted binary mask.
    """
    segmentor = TumorSegmentor(model_path=model_path)
    mask = segmentor.predict(image_path)
    
    if output_path:
        # Save mask as image
        from PIL import Image
        mask_img = (mask * 255).astype(np.uint8)
        Image.fromarray(mask_img[:, :, 0]).save(output_path)
        print(f"💾 Mask saved to: {output_path}")
    
    if visualize:
        segmentor.visualize(image_path)
    
    return mask
