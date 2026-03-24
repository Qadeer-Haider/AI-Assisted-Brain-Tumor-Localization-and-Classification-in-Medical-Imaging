"""
Segmentation metrics for training evaluation.

Provides Dice coefficient, IoU, sensitivity, specificity, precision,
and loss functions as metrics for tracking during training.
"""

from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow as tf

try:
    from keras_unet_collection import losses as kuc_losses
    KUC_LOSSES_AVAILABLE = True
except ImportError:
    KUC_LOSSES_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# SEGMENTATION METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    """
    Dice coefficient (F1 score for segmentation).
    
    Args:
        y_true: Ground truth mask.
        y_pred: Predicted mask.
        smooth: Smoothing factor.
        
    Returns:
        Dice coefficient (0 to 1, higher is better).
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def iou_score(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    """
    Intersection over Union (Jaccard Index).
    
    Args:
        y_true: Ground truth mask.
        y_pred: Predicted mask.
        smooth: Smoothing factor.
        
    Returns:
        IoU score (0 to 1, higher is better).
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def sensitivity(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    """
    Sensitivity (True Positive Rate / Recall).
    
    Measures the proportion of actual positives correctly identified.
    
    Args:
        y_true: Ground truth mask.
        y_pred: Predicted mask.
        smooth: Smoothing factor.
        
    Returns:
        Sensitivity (0 to 1, higher is better).
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    tp = K.sum(y_true_f * y_pred_f)
    fn = K.sum(y_true_f * (1 - y_pred_f))
    return (tp + smooth) / (tp + fn + smooth)


def specificity(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    """
    Specificity (True Negative Rate).
    
    Measures the proportion of actual negatives correctly identified.
    
    Args:
        y_true: Ground truth mask.
        y_pred: Predicted mask.
        smooth: Smoothing factor.
        
    Returns:
        Specificity (0 to 1, higher is better).
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    tn = K.sum((1 - y_true_f) * (1 - y_pred_f))
    fp = K.sum((1 - y_true_f) * y_pred_f)
    return (tn + smooth) / (tn + fp + smooth)


def precision_metric(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    """
    Precision (Positive Predictive Value).
    
    Measures the proportion of positive predictions that are correct.
    
    Args:
        y_true: Ground truth mask.
        y_pred: Predicted mask.
        smooth: Smoothing factor.
        
    Returns:
        Precision (0 to 1, higher is better).
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    tp = K.sum(y_true_f * y_pred_f)
    fp = K.sum((1 - y_true_f) * y_pred_f)
    return (tp + smooth) / (tp + fp + smooth)


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS AS METRICS (tracked during training)
# ═══════════════════════════════════════════════════════════════════════════════

def dice_loss_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Dice loss as metric."""
    if KUC_LOSSES_AVAILABLE:
        return kuc_losses.dice(y_true, y_pred)
    return 1.0 - dice_coefficient(y_true, y_pred)


def tversky_loss_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Tversky loss as metric."""
    if KUC_LOSSES_AVAILABLE:
        return kuc_losses.tversky(y_true, y_pred)
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    smooth = 1e-6
    alpha, beta = 0.7, 0.3
    
    tp = K.sum(y_true_f * y_pred_f)
    fn = K.sum(y_true_f * (1 - y_pred_f))
    fp = K.sum((1 - y_true_f) * y_pred_f)
    
    tversky_idx = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - tversky_idx


def focal_tversky_loss_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Focal Tversky loss as metric."""
    if KUC_LOSSES_AVAILABLE:
        return kuc_losses.focal_tversky(y_true, y_pred)
    
    gamma = 0.75
    tversky = tversky_loss_metric(y_true, y_pred)
    return K.pow(tversky, gamma)


def dice_bce_loss_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Dice + BCE combined loss as metric."""
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    bce = K.mean(bce)
    
    if KUC_LOSSES_AVAILABLE:
        dice = kuc_losses.dice(y_true, y_pred)
    else:
        dice = 1.0 - dice_coefficient(y_true, y_pred)
    
    return 0.5 * bce + 0.5 * dice


def bce_tversky_loss_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """BCE + Tversky combined loss as metric."""
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    bce = K.mean(bce)
    
    if KUC_LOSSES_AVAILABLE:
        tversky = kuc_losses.tversky(y_true, y_pred)
    else:
        tversky = tversky_loss_metric(y_true, y_pred)
    
    return 0.5 * bce + 0.5 * tversky


def get_segmentation_metrics():
    """
    Get all segmentation metrics for model compilation.
    
    Returns:
        List of metric functions.
    """
    return [
        # Segmentation metrics
        dice_coefficient,
        iou_score,
        sensitivity,
        specificity,
        precision_metric,
        # Loss functions as metrics
        dice_loss_metric,
        tversky_loss_metric,
        focal_tversky_loss_metric,
        dice_bce_loss_metric,
        bce_tversky_loss_metric,
    ]


def get_basic_metrics():
    """
    Get basic segmentation metrics (without loss metrics).
    
    Returns:
        List of metric functions.
    """
    return [
        dice_coefficient,
        iou_score,
        sensitivity,
        specificity,
        precision_metric,
    ]
