"""
Medical-focused metrics for classification.

Provides metrics important for medical imaging tasks,
particularly those that address the risk of false negatives.
"""

from typing import List

import tensorflow as tf


def get_medical_metrics(num_classes: int = 4) -> List[tf.keras.metrics.Metric]:
    """
    Get a list of metrics suitable for medical image classification.
    
    Includes:
    - Accuracy: Overall correctness
    - Recall: Sensitivity - critical for catching tumors (avoiding false negatives)
    - Precision: Specificity signal - avoiding false positives
    - AUC: Area Under ROC Curve for multi-class classification
    
    For medical imaging, Recall is especially important because:
    - Missing a tumor (False Negative) is dangerous
    - We want to minimize the chance of telling a patient "no tumor" when one exists
    
    Args:
        num_classes: Number of output classes for AUC computation.
        
    Returns:
        List of Keras metrics.
        
    Example:
        >>> metrics = get_medical_metrics(num_classes=4)
        >>> model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=metrics)
    """
    return [
        # Overall correctness
        tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
        
        # Recall across all classes (sensitivity signal)
        # CRITICAL for medical safety - we don't want to miss tumors
        tf.keras.metrics.Recall(name="overall_recall"),
        
        # Precision across all classes
        tf.keras.metrics.Precision(name="overall_precision"),
        
        # One-vs-rest AUC for multiclass
        tf.keras.metrics.AUC(
            name="auc_ovr",
            multi_label=True,
            num_labels=num_classes,
        ),
    ]


def get_basic_metrics() -> List[tf.keras.metrics.Metric]:
    """
    Get basic classification metrics.
    
    Returns:
        List with accuracy metric only.
    """
    return [
        tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
    ]


class F1Score(tf.keras.metrics.Metric):
    """
    F1 Score metric (harmonic mean of precision and recall).
    
    Useful for imbalanced datasets where both precision and recall matter.
    
    Example:
        >>> model.compile(metrics=[F1Score()])
    """
    
    def __init__(self, name: str = "f1_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
    
    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        # F1 = 2 * (precision * recall) / (precision + recall)
        return 2 * (p * r) / (p + r + tf.keras.backend.epsilon())
    
    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()


class PerClassRecall(tf.keras.metrics.Metric):
    """
    Per-class recall for monitoring individual class performance.
    
    Useful for identifying which tumor types the model struggles with.
    
    Args:
        class_idx: Index of the class to monitor.
        class_name: Name of the class for display.
    """
    
    def __init__(
        self,
        class_idx: int,
        class_name: str = None,
        name: str = None,
        **kwargs
    ):
        if name is None:
            name = f"recall_class_{class_idx}" if class_name is None else f"recall_{class_name}"
        super().__init__(name=name, **kwargs)
        
        self.class_idx = class_idx
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.possible_positives = self.add_weight(name="pp", initializer="zeros")
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Get class predictions and labels
        y_true_class = y_true[:, self.class_idx]
        y_pred_class = tf.cast(
            tf.argmax(y_pred, axis=-1) == self.class_idx,
            tf.float32
        )
        
        # Update counts
        self.true_positives.assign_add(
            tf.reduce_sum(y_true_class * y_pred_class)
        )
        self.possible_positives.assign_add(
            tf.reduce_sum(y_true_class)
        )
    
    def result(self):
        return self.true_positives / (self.possible_positives + tf.keras.backend.epsilon())
    
    def reset_state(self):
        self.true_positives.assign(0)
        self.possible_positives.assign(0)
