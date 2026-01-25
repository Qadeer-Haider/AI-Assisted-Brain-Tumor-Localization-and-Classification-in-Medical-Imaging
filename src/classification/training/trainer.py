"""
Classification trainer module.

Provides a high-level training interface for brain tumor classification.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import tensorflow as tf

from ...utils.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    NUM_CLASSES,
    WEIGHTS_DIR,
)
from ..data import build_dataframe, compute_class_weights, get_stratified_split, make_dataset
from ..data.preprocessing import get_train_augmentation
from ..models import build_model
from .callbacks import get_callbacks
from .metrics import get_medical_metrics


class ClassificationTrainer:
    """
    High-level trainer for brain tumor classification models.
    
    Handles the complete training pipeline:
    - Data loading and preprocessing
    - Model building
    - Training with callbacks
    - Evaluation
    
    Attributes:
        model: The Keras model being trained.
        history: Training history after fit().
        config: Configuration dictionary.
        
    Example:
        >>> trainer = ClassificationTrainer(model_name="ConvNeXtBase")
        >>> trainer.prepare_data()
        >>> trainer.train(epochs=100)
        >>> trainer.evaluate()
    """
    
    def __init__(
        self,
        model_name: str = "ConvNeXtBase",
        num_classes: int = NUM_CLASSES,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            model_name: Name of the backbone architecture.
            num_classes: Number of output classes.
            batch_size: Training batch size.
            learning_rate: Initial learning rate.
            config: Optional configuration dictionary.
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.config = config or {}
        
        # Model and history
        self.model: Optional[tf.keras.Model] = None
        self.history: Optional[tf.keras.callbacks.History] = None
        
        # Datasets
        self.train_ds: Optional[tf.data.Dataset] = None
        self.val_ds: Optional[tf.data.Dataset] = None
        self.test_ds: Optional[tf.data.Dataset] = None
        
        # DataFrames (for later access)
        self.train_df: Optional[Any] = None
        self.val_df: Optional[Any] = None
        self.test_df: Optional[Any] = None
        
        # Class weights
        self.class_weights: Optional[Dict[int, float]] = None
        
        # Paths (set during training)
        self.checkpoint_path: Optional[Path] = None
        self.log_path: Optional[Path] = None
    
    def prepare_data(
        self,
        train_dir: Optional[Path] = None,
        test_dir: Optional[Path] = None,
        val_split: float = 0.2,
        use_augmentation: bool = True,
    ) -> None:
        """
        Prepare training, validation, and test datasets.
        
        Args:
            train_dir: Path to training data directory.
            test_dir: Path to test data directory.
            val_split: Fraction of training data for validation.
            use_augmentation: Whether to use data augmentation.
        """
        print("📂 Preparing datasets...")
        
        # Build DataFrames
        train_val_df = build_dataframe(train_dir, split="train")
        test_df = build_dataframe(test_dir, split="test")
        
        # Split train/val
        train_df, val_df = get_stratified_split(train_val_df, val_split=val_split)
        
        print(f"   ↳ Training samples: {len(train_df)}")
        print(f"   ↳ Validation samples: {len(val_df)}")
        print(f"   ↳ Test samples: {len(test_df)}")
        
        # Compute class weights
        self.class_weights = compute_class_weights(train_df)
        print(f"   ↳ Class weights: {self.class_weights}")
        
        # Get augmentation
        train_aug = get_train_augmentation() if use_augmentation else None
        
        # Create datasets
        self.train_ds = make_dataset(
            train_df,
            model_name=self.model_name,
            shuffle=True,
            batch_size=self.batch_size,
            augmentation=train_aug,
        )
        
        self.val_ds = make_dataset(
            val_df,
            model_name=self.model_name,
            shuffle=False,
            batch_size=self.batch_size,
        )
        
        self.test_ds = make_dataset(
            test_df,
            model_name=self.model_name,
            shuffle=False,
            batch_size=self.batch_size,
        )
        
        # Store DataFrames
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        
        print("✅ Datasets prepared successfully!")
    
    def build(self) -> tf.keras.Model:
        """
        Build the classification model.
        
        Returns:
            Built Keras model.
        """
        self.model = build_model(
            model_name=self.model_name,
            num_classes=self.num_classes,
        )
        return self.model
    
    def compile(
        self,
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        loss: str = "categorical_crossentropy",
        metrics: Optional[list] = None,
    ) -> None:
        """
        Compile the model.
        
        Args:
            optimizer: Keras optimizer. If None, uses Adam with configured LR.
            loss: Loss function name.
            metrics: List of metrics. If None, uses medical metrics.
        """
        if self.model is None:
            self.build()
        
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        if metrics is None:
            metrics = get_medical_metrics(self.num_classes)
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
        )
        
        print("✅ Model compiled successfully!")
    
    def train(
        self,
        epochs: int = DEFAULT_EPOCHS,
        callbacks: Optional[list] = None,
        use_class_weights: bool = True,
    ) -> tf.keras.callbacks.History:
        """
        Train the model.
        
        Args:
            epochs: Number of training epochs.
            callbacks: List of Keras callbacks. If None, uses default callbacks.
            use_class_weights: Whether to use class weights.
            
        Returns:
            Training history.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() or compile() first.")
        
        if self.train_ds is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        if callbacks is None:
            callback_list = get_callbacks(self.model_name)
            # Extract checkpoint path from callbacks for later reference
            for cb in callback_list:
                if hasattr(cb, 'filepath'):
                    self.checkpoint_path = Path(cb.filepath)
        else:
            callback_list = callbacks
        
        # Prepare class weights
        class_weight = self.class_weights if use_class_weights else None
        
        print(f"\n🚀 Starting training for {self.model_name}...")
        print(f"   ↳ Epochs: {epochs}")
        print(f"   ↳ Using class weights: {use_class_weights}")
        
        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            callbacks=callback_list,
            class_weight=class_weight,
        )
        
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
            dataset = self.test_ds
        
        if dataset is None:
            raise ValueError("No dataset provided and test_ds not prepared.")
        
        print("\n📊 Evaluating model...")
        
        results = self.model.evaluate(dataset, verbose=1, return_dict=True)
        
        print("\n📊 Test Metrics:")
        for name, value in results.items():
            if name in ["accuracy", "overall_recall", "overall_precision", "auc_ovr"]:
                print(f"   ↳ {name}: {value*100:.2f}%")
            else:
                print(f"   ↳ {name}: {value:.4f}")
        
        return results
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save the model.
        
        Args:
            path: Path to save the model. If None, uses default location.
        """
        if self.model is None:
            raise ValueError("No model to save.")
        
        if path is None:
            path = WEIGHTS_DIR / "classification" / f"{self.model_name}_final.keras"
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save(str(path))
        print(f"✅ Model saved to: {path}")
    
    def load(self, path: str) -> tf.keras.Model:
        """
        Load a saved model.
        
        Args:
            path: Path to the saved model.
            
        Returns:
            Loaded Keras model.
        """
        self.model = tf.keras.models.load_model(path)
        print(f"✅ Model loaded from: {path}")
        return self.model


def train_model(
    model_name: str = "ConvNeXtBase",
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    **kwargs,
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Convenience function to train a classification model.
    
    Args:
        model_name: Name of the backbone architecture.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Initial learning rate.
        **kwargs: Additional arguments passed to ClassificationTrainer.
        
    Returns:
        Tuple of (trained model, training history).
        
    Example:
        >>> model, history = train_model("ConvNeXtBase", epochs=100)
    """
    trainer = ClassificationTrainer(
        model_name=model_name,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    
    trainer.prepare_data(**kwargs)
    trainer.build()
    trainer.compile()
    history = trainer.train(epochs=epochs)
    
    return trainer.model, history
