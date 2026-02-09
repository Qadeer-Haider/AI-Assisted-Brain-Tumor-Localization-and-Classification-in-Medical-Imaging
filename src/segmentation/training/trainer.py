"""
Segmentation trainer module.

Provides a high-level training interface for brain tumor segmentation.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import tensorflow as tf

from ..data import create_segmentation_datasets
from ..models import build_segmentation_model
from .callbacks import get_callbacks
from .losses import get_loss_function
from .metrics import get_segmentation_metrics


class SegmentationTrainer:
    """
    High-level trainer for brain tumor segmentation models.
    
    Handles the complete training pipeline:
    - Data loading with image-mask pairs
    - Model building
    - Training with appropriate losses
    - Evaluation
    
    Example:
        >>> trainer = SegmentationTrainer(model_name="UNet", loss_name="bce_tversky")
        >>> trainer.prepare_data("data/train/images", "data/train/masks")
        >>> trainer.build()
        >>> trainer.compile()
        >>> trainer.train(epochs=100)
        >>> trainer.evaluate()
    """
    
    def __init__(
        self,
        model_name: str = "UNet",
        loss_name: str = "bce_tversky",
        num_classes: int = 1,
        img_size: Tuple[int, int] = (256, 256),
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        backbone: str = "ResNet50",
        freeze_backbone: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            model_name: Name of the segmentation architecture.
                Options: "UNet", "AttentionUNet", "ResUNetPP", "SwinUNet"
            loss_name: Name of the loss function.
                Options: "dice", "dice_bce", "tversky", "focal_tversky", "bce_tversky"
            num_classes: Number of output classes (1 for binary).
            img_size: Target image size (height, width).
            batch_size: Training batch size.
            learning_rate: Initial learning rate.
            backbone: Backbone architecture for encoder.
            freeze_backbone: Whether to freeze backbone weights.
            config: Optional configuration dictionary (overrides other args).
        """
        # Apply config if provided
        if config:
            model_name = config.get("model_name", model_name)
            loss_name = config.get("loss", loss_name)
            num_classes = config.get("num_classes", num_classes)
            img_size = tuple(config.get("img_size", img_size))
            batch_size = config.get("batch_size", batch_size)
            learning_rate = config.get("learning_rate", learning_rate)
            backbone = config.get("backbone", backbone)
            freeze_backbone = config.get("freeze_backbone", freeze_backbone)
        
        self.model_name = model_name
        self.loss_name = loss_name
        self.num_classes = num_classes
        self.img_size = img_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        self.config = config or {}
        
        self.model: Optional[tf.keras.Model] = None
        self.history: Optional[tf.keras.callbacks.History] = None
        self.train_ds: Optional[tf.data.Dataset] = None
        self.val_ds: Optional[tf.data.Dataset] = None
        self.test_ds: Optional[tf.data.Dataset] = None
    
    def prepare_data(
        self,
        train_images_dir: str,
        train_masks_dir: str,
        test_images_dir: Optional[str] = None,
        test_masks_dir: Optional[str] = None,
        val_split: float = 0.2,
        random_state: int = 42,
        use_augmentation: bool = True,
    ):
        """
        Prepare training, validation, and test datasets.
        
        Args:
            train_images_dir: Directory containing training images.
            train_masks_dir: Directory containing training masks.
            test_images_dir: Optional directory for test images.
            test_masks_dir: Optional directory for test masks.
            val_split: Fraction of training data for validation.
            random_state: Random seed for reproducibility.
            use_augmentation: Whether to apply augmentation.
        """
        print("\n📂 Preparing datasets...")
        
        self.train_ds, self.val_ds, self.test_ds = create_segmentation_datasets(
            train_images_dir=train_images_dir,
            train_masks_dir=train_masks_dir,
            test_images_dir=test_images_dir,
            test_masks_dir=test_masks_dir,
            img_size=self.img_size,
            batch_size=self.batch_size,
            val_split=val_split,
            random_state=random_state,
            use_augmentation=use_augmentation,
        )
        
        print("✅ Datasets prepared successfully!")
    
    def build(self) -> tf.keras.Model:
        """
        Build the segmentation model.
        
        Returns:
            Built Keras model.
        """
        input_shape = (*self.img_size, 3)
        
        self.model = build_segmentation_model(
            model_name=self.model_name,
            input_shape=input_shape,
            num_classes=self.num_classes,
            backbone=self.backbone,
            freeze_backbone=self.freeze_backbone,
        )
        
        return self.model
    
    def compile(
        self,
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        metrics: Optional[list] = None,
    ):
        """
        Compile the model with loss and metrics.
        
        Args:
            optimizer: Keras optimizer. If None, uses Adam.
            metrics: List of metrics. If None, uses segmentation metrics.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        if metrics is None:
            metrics = get_segmentation_metrics()
        
        loss_fn = get_loss_function(self.loss_name)
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metrics,
        )
        
        print(f"\n✅ Model compiled with {self.loss_name} loss")
        print(f"   All loss functions tracked as metrics during training")
    
    def train(
        self,
        epochs: int = 100,
        callbacks: Optional[list] = None,
        weights_dir: str = "weights/segmentation",
        logs_dir: str = "logs/segmentation",
    ) -> tf.keras.callbacks.History:
        """
        Train the model.
        
        Args:
            epochs: Number of training epochs.
            callbacks: List of Keras callbacks. If None, uses default callbacks.
            weights_dir: Directory for saving model weights.
            logs_dir: Directory for training logs.
            
        Returns:
            Training history.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        if self.train_ds is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        # Default callbacks
        if callbacks is None:
            checkpoint_path = f"{weights_dir}/{self.model_name}_{self.loss_name}_best.keras"
            log_path = f"{logs_dir}/{self.model_name}_{self.loss_name}_training_log.csv"
            
            callbacks = get_callbacks(
                checkpoint_path=checkpoint_path,
                log_path=log_path,
            )
        
        print(f"\n🚀 Training {self.model_name} with {self.loss_name} loss...")
        
        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )
        
        print("\n✅ Training completed!")
        
        return self.history
    
    def evaluate(self, dataset: Optional[tf.data.Dataset] = None) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataset: Dataset to evaluate on. If None, uses test set.
            
        Returns:
            Dictionary of metric names and values.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        if dataset is None:
            if self.test_ds is not None:
                dataset = self.test_ds
            elif self.val_ds is not None:
                dataset = self.val_ds
            else:
                raise ValueError("No dataset to evaluate on.")
        
        print("\n📊 Evaluating model...")
        results = self.model.evaluate(dataset, verbose=1)
        
        # Create results dictionary
        metric_names = ["loss"] + [m.name if hasattr(m, "name") else str(m) for m in self.model.metrics]
        results_dict = dict(zip(metric_names[:len(results)], results))
        
        return results_dict
    
    def save(self, path: Optional[str] = None):
        """
        Save the model.
        
        Args:
            path: Path to save the model. If None, uses default location.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        if path is None:
            path = f"weights/segmentation/{self.model_name}_{self.loss_name}_final.keras"
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        print(f"💾 Model saved to: {path}")
    
    def load(self, path: str) -> tf.keras.Model:
        """
        Load a saved model.
        
        Args:
            path: Path to the saved model.
            
        Returns:
            Loaded Keras model.
        """
        # Custom objects for loading
        custom_objects = {}
        
        # Add metrics
        from .metrics import (
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
        
        # Add loss functions
        from .losses import LOSS_FUNCTIONS
        custom_objects.update(LOSS_FUNCTIONS)
        
        self.model = tf.keras.models.load_model(path, custom_objects=custom_objects)
        print(f"📂 Model loaded from: {path}")
        
        return self.model


def train_segmentation_model(
    model_name: str = "UNet",
    loss_name: str = "bce_tversky",
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    train_images_dir: str = "data/brisc2025/segmentation_task/train/images",
    train_masks_dir: str = "data/brisc2025/segmentation_task/train/masks",
    test_images_dir: Optional[str] = "data/brisc2025/segmentation_task/test/images",
    test_masks_dir: Optional[str] = "data/brisc2025/segmentation_task/test/masks",
    **kwargs,
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Convenience function to train a segmentation model.
    
    Args:
        model_name: Name of the backbone architecture.
        loss_name: Name of the loss function.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Initial learning rate.
        train_images_dir: Path to training images.
        train_masks_dir: Path to training masks.
        test_images_dir: Path to test images.
        test_masks_dir: Path to test masks.
        **kwargs: Additional arguments passed to trainer.
        
    Returns:
        Tuple of (trained model, training history).
    """
    trainer = SegmentationTrainer(
        model_name=model_name,
        loss_name=loss_name,
        batch_size=batch_size,
        learning_rate=learning_rate,
        **kwargs,
    )
    
    trainer.prepare_data(
        train_images_dir=train_images_dir,
        train_masks_dir=train_masks_dir,
        test_images_dir=test_images_dir,
        test_masks_dir=test_masks_dir,
    )
    
    trainer.build()
    trainer.compile()
    history = trainer.train(epochs=epochs)
    
    return trainer.model, history
