"""
Visualization utilities for training and inference.

Provides plotting functions for training curves, predictions,
confusion matrices, and sample images.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from .constants import CLASS_NAMES


def plot_training_history(
    history: Dict[str, List[float]],
    metrics: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> None:
    """
    Plot training history curves.
    
    Args:
        history: Dictionary with training metrics (e.g., from model.fit()).
        metrics: List of metrics to plot. If None, plots loss and accuracy.
        save_path: Optional path to save the figure.
        figsize: Figure size (width, height).
    """
    if metrics is None:
        metrics = ["loss", "accuracy"]
    
    # Filter to available metrics
    available = [m for m in metrics if m in history or f"val_{m}" in history]
    
    fig, axes = plt.subplots(1, len(available), figsize=figsize)
    if len(available) == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, available):
        if metric in history:
            ax.plot(history[metric], label=f"Train {metric}", linewidth=2)
        if f"val_{metric}" in history:
            ax.plot(history[f"val_{metric}"], label=f"Val {metric}", linewidth=2)
        
        ax.set_title(f"{metric.replace('_', ' ').title()}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📊 Plot saved to: {save_path}")
    
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = "Blues",
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix array.
        class_names: List of class names. If None, uses default CLASS_NAMES.
        normalize: Whether to normalize the matrix.
        save_path: Optional path to save the figure.
        figsize: Figure size (width, height).
        cmap: Colormap for the heatmap.
    """
    if class_names is None:
        class_names = CLASS_NAMES
    
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2%"
    else:
        fmt = "d"
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = f"{cm[i, j]:{fmt}}" if normalize else f"{cm[i, j]}"
            ax.text(
                j, i, value,
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📊 Confusion matrix saved to: {save_path}")
    
    plt.show()


def plot_prediction(
    image: np.ndarray,
    predicted_class: str,
    confidence: float,
    true_class: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a single prediction with image and result.
    
    Args:
        image: Image array (H, W, C) normalized to [0, 1] or [0, 255].
        predicted_class: Predicted class name.
        confidence: Prediction confidence (0-100).
        true_class: Optional true class name for comparison.
        save_path: Optional path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Normalize image for display if needed
    if image.max() > 1:
        image = image / 255.0
    
    ax.imshow(image)
    ax.axis("off")
    
    # Build title
    title = f"Predicted: {predicted_class} ({confidence:.1f}%)"
    if true_class:
        correct = "✓" if predicted_class == true_class else "✗"
        title += f"\nTrue: {true_class} {correct}"
        color = "green" if predicted_class == true_class else "red"
    else:
        color = "blue"
    
    ax.set_title(title, fontsize=12, fontweight="bold", color=color)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    plt.show()


def plot_sample_predictions(
    images: List[np.ndarray],
    predictions: List[str],
    confidences: List[float],
    true_labels: Optional[List[str]] = None,
    n_cols: int = 4,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a grid of predictions.
    
    Args:
        images: List of image arrays.
        predictions: List of predicted class names.
        confidences: List of confidence values.
        true_labels: Optional list of true labels.
        n_cols: Number of columns in the grid.
        save_path: Optional path to save the figure.
    """
    n_samples = len(images)
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = axes.flatten() if n_samples > 1 else [axes]
    
    for idx, ax in enumerate(axes):
        if idx < n_samples:
            image = images[idx]
            if image.max() > 1:
                image = image / 255.0
            
            ax.imshow(image)
            ax.axis("off")
            
            pred = predictions[idx]
            conf = confidences[idx]
            title = f"{pred}\n{conf:.1f}%"
            
            if true_labels:
                true = true_labels[idx]
                color = "green" if pred == true else "red"
            else:
                color = "black"
            
            ax.set_title(title, fontsize=10, color=color)
        else:
            ax.axis("off")
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📊 Sample predictions saved to: {save_path}")
    
    plt.show()
