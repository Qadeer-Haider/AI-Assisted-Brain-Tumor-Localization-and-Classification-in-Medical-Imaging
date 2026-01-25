"""
Model evaluation utilities for classification.

Provides comprehensive evaluation including confusion matrices,
per-class metrics, and detailed reports.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from ...utils.constants import CLASS_NAMES, NUM_CLASSES
from ...utils.visualization import plot_confusion_matrix


class ClassificationEvaluator:
    """
    Comprehensive evaluator for classification models.
    
    Provides detailed metrics including:
    - Overall accuracy, precision, recall
    - Per-class performance
    - Confusion matrix
    - Detailed classification report
    
    Example:
        >>> evaluator = ClassificationEvaluator(model, test_ds)
        >>> results = evaluator.evaluate()
        >>> evaluator.plot_confusion_matrix()
    """
    
    def __init__(
        self,
        model: tf.keras.Model,
        dataset: tf.data.Dataset,
        class_names: Optional[List[str]] = None,
    ):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained Keras model.
            dataset: Test dataset (tf.data.Dataset).
            class_names: List of class names.
        """
        self.model = model
        self.dataset = dataset
        self.class_names = class_names or CLASS_NAMES
        self.num_classes = len(self.class_names)
        
        # Cached predictions
        self._y_true: Optional[np.ndarray] = None
        self._y_pred: Optional[np.ndarray] = None
        self._y_pred_proba: Optional[np.ndarray] = None
    
    def _get_predictions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get predictions and true labels from the dataset.
        
        Returns:
            Tuple of (y_true, y_pred, y_pred_proba).
        """
        if self._y_true is not None:
            return self._y_true, self._y_pred, self._y_pred_proba
        
        y_true_list = []
        y_pred_proba_list = []
        
        for images, labels in self.dataset:
            predictions = self.model.predict(images, verbose=0)
            y_pred_proba_list.append(predictions)
            y_true_list.append(labels.numpy())
        
        self._y_true = np.concatenate(y_true_list, axis=0)
        self._y_pred_proba = np.concatenate(y_pred_proba_list, axis=0)
        self._y_pred = np.argmax(self._y_pred_proba, axis=1)
        
        # Convert one-hot to class indices
        self._y_true = np.argmax(self._y_true, axis=1)
        
        return self._y_true, self._y_pred, self._y_pred_proba
    
    def evaluate(self, verbose: bool = True) -> Dict[str, float]:
        """
        Run evaluation and return metrics.
        
        Args:
            verbose: Whether to print results.
            
        Returns:
            Dictionary of metric names and values.
        """
        # Use Keras evaluate for standard metrics
        results = self.model.evaluate(self.dataset, verbose=1, return_dict=True)
        
        # Get predictions for additional analysis
        y_true, y_pred, _ = self._get_predictions()
        
        # Compute additional metrics
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
        )
        
        results["sklearn_accuracy"] = accuracy_score(y_true, y_pred)
        results["macro_precision"] = precision_score(y_true, y_pred, average="macro")
        results["macro_recall"] = recall_score(y_true, y_pred, average="macro")
        results["macro_f1"] = f1_score(y_true, y_pred, average="macro")
        results["weighted_f1"] = f1_score(y_true, y_pred, average="weighted")
        
        if verbose:
            print("\n📊 Evaluation Results:")
            print("=" * 50)
            for name, value in results.items():
                if any(x in name for x in ["accuracy", "recall", "precision", "f1", "auc"]):
                    print(f"   {name}: {value*100:.2f}%")
                else:
                    print(f"   {name}: {value:.4f}")
        
        return results
    
    def get_confusion_matrix(self) -> np.ndarray:
        """
        Compute confusion matrix.
        
        Returns:
            Confusion matrix array.
        """
        from sklearn.metrics import confusion_matrix
        
        y_true, y_pred, _ = self._get_predictions()
        return confusion_matrix(y_true, y_pred)
    
    def plot_confusion_matrix(
        self,
        normalize: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot the confusion matrix.
        
        Args:
            normalize: Whether to normalize the matrix.
            save_path: Optional path to save the figure.
        """
        cm = self.get_confusion_matrix()
        plot_confusion_matrix(
            cm,
            class_names=self.class_names,
            normalize=normalize,
            save_path=save_path,
        )
    
    def get_per_class_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get per-class precision, recall, and F1 scores.
        
        Returns:
            Dictionary with metrics for each class.
        """
        from sklearn.metrics import precision_recall_fscore_support
        
        y_true, y_pred, _ = self._get_predictions()
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(self.num_classes)
        )
        
        results = {}
        for i, name in enumerate(self.class_names):
            results[name] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
        
        return results
    
    def print_classification_report(self) -> str:
        """
        Print and return sklearn's classification report.
        
        Returns:
            Classification report string.
        """
        from sklearn.metrics import classification_report
        
        y_true, y_pred, _ = self._get_predictions()
        
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            digits=4,
        )
        
        print("\n📋 Classification Report:")
        print("=" * 60)
        print(report)
        
        return report


def evaluate_model(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    class_names: Optional[List[str]] = None,
    plot_cm: bool = True,
) -> Dict[str, float]:
    """
    Convenience function to evaluate a model.
    
    Args:
        model: Trained Keras model.
        dataset: Test dataset.
        class_names: List of class names.
        plot_cm: Whether to plot confusion matrix.
        
    Returns:
        Dictionary of metrics.
        
    Example:
        >>> results = evaluate_model(model, test_ds)
    """
    evaluator = ClassificationEvaluator(model, dataset, class_names)
    results = evaluator.evaluate()
    
    if plot_cm:
        evaluator.plot_confusion_matrix()
    
    evaluator.print_classification_report()
    
    return results
