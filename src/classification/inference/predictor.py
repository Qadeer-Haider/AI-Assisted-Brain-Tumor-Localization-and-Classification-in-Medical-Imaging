"""
Tumor classification predictor.

Provides inference utilities for single images and batch predictions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image

from ...utils.constants import CLASS_NAMES, IDX_TO_CLASS, IMG_SIZE
from ..data.preprocessing import get_preprocessing_fn


class TumorClassifier:
    """
    High-level interface for brain tumor classification inference.
    
    Provides methods for single image prediction, batch prediction,
    and confidence analysis.
    
    Attributes:
        model: Loaded Keras model.
        class_names: List of class names.
        img_size: Expected input image size.
        
    Example:
        >>> classifier = TumorClassifier("path/to/model.keras")
        >>> result = classifier.predict("path/to/image.jpg")
        >>> print(result)
        {'class': 'glioma', 'confidence': 95.5, 'probabilities': {...}}
    """
    
    def __init__(
        self,
        model_path: str,
        class_names: Optional[List[str]] = None,
        img_size: Tuple[int, int] = IMG_SIZE,
        model_name: str = "ConvNeXtBase",
    ):
        """
        Initialize the classifier.
        
        Args:
            model_path: Path to the saved Keras model.
            class_names: List of class names. If None, uses default.
            img_size: Expected input image size.
            model_name: Name of the model architecture (for preprocessing).
        """
        self.model_path = Path(model_path)
        self.class_names = class_names or CLASS_NAMES
        self.img_size = img_size
        self.model_name = model_name
        
        # Load model
        print(f"📂 Loading model from {self.model_path}...")
        self.model = tf.keras.models.load_model(str(self.model_path))
        print("✅ Model loaded successfully!")
        
        # Get preprocessing function
        self.preprocess_fn = get_preprocessing_fn(model_name)
    
    def predict(
        self,
        image_path: str,
        return_probabilities: bool = True,
    ) -> Dict[str, any]:
        """
        Predict the class of a single image.
        
        Args:
            image_path: Path to the image file.
            return_probabilities: Whether to include all class probabilities.
            
        Returns:
            Dictionary with prediction results:
            - 'class': Predicted class name
            - 'confidence': Confidence percentage
            - 'probabilities': Dict of all class probabilities (optional)
            - 'warning': Medical warning message
        """
        # Load and preprocess image
        img = keras_image.load_img(image_path, target_size=self.img_size)
        img_array = keras_image.img_to_array(img)
        img_array = self.preprocess_fn(img_array)
        img_batch = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = self.model.predict(img_batch, verbose=0)
        
        # Get top prediction
        idx = np.argmax(predictions[0])
        predicted_class = self.class_names[idx]
        confidence = float(predictions[0][idx] * 100)
        
        result = {
            "class": predicted_class,
            "confidence": confidence,
            "image_path": str(image_path),
        }
        
        # Add all probabilities if requested
        if return_probabilities:
            result["probabilities"] = {
                name: float(predictions[0][i] * 100)
                for i, name in enumerate(self.class_names)
            }
        
        # Add medical warning
        if predicted_class == "no_tumor":
            result["warning"] = "ℹ️ Model predicts NO TUMOR — verify recall & clinical context."
        else:
            result["warning"] = "⚠️ TUMOR DETECTED — requires clinical confirmation."
        
        return result
    
    def predict_batch(
        self,
        image_paths: List[str],
        batch_size: int = 32,
    ) -> List[Dict[str, any]]:
        """
        Predict classes for a batch of images.
        
        Args:
            image_paths: List of image paths.
            batch_size: Batch size for prediction.
            
        Returns:
            List of prediction dictionaries.
        """
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            # Load and preprocess batch
            batch_images = []
            for path in batch_paths:
                img = keras_image.load_img(path, target_size=self.img_size)
                img_array = keras_image.img_to_array(img)
                img_array = self.preprocess_fn(img_array)
                batch_images.append(img_array)
            
            batch_array = np.array(batch_images)
            
            # Predict
            predictions = self.model.predict(batch_array, verbose=0)
            
            # Process each prediction
            for j, path in enumerate(batch_paths):
                idx = np.argmax(predictions[j])
                results.append({
                    "class": self.class_names[idx],
                    "confidence": float(predictions[j][idx] * 100),
                    "image_path": str(path),
                })
        
        return results
    
    def get_top_k_predictions(
        self,
        image_path: str,
        k: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Get top-k predictions with confidence scores.
        
        Args:
            image_path: Path to the image file.
            k: Number of top predictions to return.
            
        Returns:
            List of (class_name, confidence) tuples.
        """
        result = self.predict(image_path, return_probabilities=True)
        probs = result["probabilities"]
        
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_probs[:k]


def predict_single_image(
    image_path: str,
    model: tf.keras.Model,
    class_names: Optional[List[str]] = None,
    img_size: Tuple[int, int] = IMG_SIZE,
    preprocess_fn: Optional[callable] = None,
    show_image: bool = True,
) -> Dict[str, any]:
    """
    Predict the class of a single image and optionally display it.
    
    This is a standalone function that matches the notebook's
    predict_single_image() function.
    
    Args:
        image_path: Path to the image file.
        model: Loaded Keras model.
        class_names: List of class names.
        img_size: Target image size.
        preprocess_fn: Preprocessing function. If None, uses basic scaling.
        show_image: Whether to display the image with prediction.
        
    Returns:
        Dictionary with prediction results.
        
    Example:
        >>> model = tf.keras.models.load_model("model.keras")
        >>> result = predict_single_image("image.jpg", model)
    """
    import matplotlib.pyplot as plt
    
    if class_names is None:
        class_names = CLASS_NAMES
    
    # Load and preprocess
    img = keras_image.load_img(image_path, target_size=img_size)
    img_array = keras_image.img_to_array(img)
    
    if preprocess_fn is not None:
        img_array_processed = preprocess_fn(img_array.copy())
    else:
        img_array_processed = img_array / 255.0
    
    img_batch = np.expand_dims(img_array_processed, axis=0)
    
    # Predict
    preds = model.predict(img_batch, verbose=0)
    idx = np.argmax(preds[0])
    predicted_class = class_names[idx]
    confidence = preds[0][idx] * 100
    
    # Display image
    if show_image:
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"{predicted_class} ({confidence:.2f}%)")
        plt.show()
    
    # Print results
    print(f"✅ Prediction: {predicted_class}")
    print(f"📊 Confidence: {confidence:.2f}%")
    
    # Medical-style warning
    if predicted_class == "no_tumor":
        print("ℹ️ Model predicts NO TUMOR — verify recall & clinical context.")
    else:
        print("⚠️ TUMOR DETECTED — requires clinical confirmation.")
    
    return {
        "class": predicted_class,
        "confidence": confidence,
        "probabilities": {
            name: float(preds[0][i] * 100)
            for i, name in enumerate(class_names)
        },
    }
