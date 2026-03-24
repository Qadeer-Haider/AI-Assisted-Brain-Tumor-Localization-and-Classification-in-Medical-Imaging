"""
Loss functions for segmentation.

Provides Dice loss, BCE loss, Tversky loss, and combined losses for
binary segmentation tasks.
"""

from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow as tf

try:
    from keras_unet_collection import losses as kuc_losses
    KUC_LOSSES_AVAILABLE = True
except ImportError:
    KUC_LOSSES_AVAILABLE = False


def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    """
    Compute Dice coefficient (F1 score for segmentation).
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    
    Args:
        y_true: Ground truth mask.
        y_pred: Predicted mask.
        smooth: Smoothing factor to avoid division by zero.
        
    Returns:
        Dice coefficient (0 to 1, higher is better).
    """
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    
    intersection = K.sum(y_true_flat * y_pred_flat)
    union = K.sum(y_true_flat) + K.sum(y_pred_flat)
    
    return (2.0 * intersection + smooth) / (union + smooth)


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Dice loss = 1 - Dice coefficient.
    
    Optimizing this loss directly maximizes the Dice score.
    
    Args:
        y_true: Ground truth mask.
        y_pred: Predicted mask.
        
    Returns:
        Dice loss (0 to 1, lower is better).
    """
    if KUC_LOSSES_AVAILABLE:
        return kuc_losses.dice(y_true, y_pred)
    return 1.0 - dice_coefficient(y_true, y_pred)


def dice_bce_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
) -> tf.Tensor:
    """
    Combined Binary Cross-Entropy and Dice loss.
    
    This combination often works better than either alone:
    - BCE provides pixel-wise gradients
    - Dice provides region-level optimization
    
    Args:
        y_true: Ground truth mask.
        y_pred: Predicted mask.
        bce_weight: Weight for BCE loss.
        dice_weight: Weight for Dice loss.
        
    Returns:
        Combined loss.
    """
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    bce = K.mean(bce)
    
    if KUC_LOSSES_AVAILABLE:
        dice = kuc_losses.dice(y_true, y_pred)
    else:
        dice = dice_loss(y_true, y_pred)
    
    return bce_weight * bce + dice_weight * dice


def tversky_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    alpha: float = 0.7,
    beta: float = 0.3,
    smooth: float = 1e-6,
) -> tf.Tensor:
    """
    Tversky loss for imbalanced segmentation.
    
    Generalizes Dice loss with adjustable FP/FN penalties.
    alpha > beta: penalize false negatives more (good for tumor detection)
    
    Args:
        y_true: Ground truth mask.
        y_pred: Predicted mask.
        alpha: Weight for false positives.
        beta: Weight for false negatives.
        smooth: Smoothing factor.
        
    Returns:
        Tversky loss.
    """
    if KUC_LOSSES_AVAILABLE:
        return kuc_losses.tversky(y_true, y_pred)
    
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    
    true_pos = K.sum(y_true_flat * y_pred_flat)
    false_neg = K.sum(y_true_flat * (1 - y_pred_flat))
    false_pos = K.sum((1 - y_true_flat) * y_pred_flat)
    
    tversky_index = (true_pos + smooth) / (
        true_pos + alpha * false_pos + beta * false_neg + smooth
    )
    
    return 1.0 - tversky_index


def focal_tversky_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    gamma: float = 0.75,
) -> tf.Tensor:
    """
    Focal Tversky loss for hard regions focus.
    
    Applies focal weighting to Tversky loss to focus on difficult regions.
    
    Args:
        y_true: Ground truth mask.
        y_pred: Predicted mask.
        gamma: Focal parameter.
        
    Returns:
        Focal Tversky loss.
    """
    if KUC_LOSSES_AVAILABLE:
        return kuc_losses.focal_tversky(y_true, y_pred)
    
    tversky = tversky_loss(y_true, y_pred)
    return K.pow(tversky, gamma)


def bce_tversky_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    bce_weight: float = 0.5,
    tversky_weight: float = 0.5,
) -> tf.Tensor:
    """
    BCE + Tversky combined loss (DEFAULT - balanced and robust).
    
    Combines the stability of BCE with the region-awareness of Tversky.
    
    Args:
        y_true: Ground truth mask.
        y_pred: Predicted mask.
        bce_weight: Weight for BCE loss.
        tversky_weight: Weight for Tversky loss.
        
    Returns:
        Combined loss.
    """
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    bce = K.mean(bce)
    
    if KUC_LOSSES_AVAILABLE:
        tversky = kuc_losses.tversky(y_true, y_pred)
    else:
        tversky = tversky_loss(y_true, y_pred)
    
    return bce_weight * bce + tversky_weight * tversky


def focal_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> tf.Tensor:
    """
    Focal loss for handling class imbalance in segmentation.
    
    Focuses learning on hard, misclassified pixels.
    
    Args:
        y_true: Ground truth mask.
        y_pred: Predicted mask.
        gamma: Focusing parameter.
        alpha: Class balance weight.
        
    Returns:
        Focal loss.
    """
    # Clip predictions to avoid log(0)
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    
    # Compute cross-entropy
    ce = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
    
    # Compute focal weight
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    focal_weight = K.pow(1 - p_t, gamma)
    
    # Apply alpha weighting
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    
    focal = alpha_t * focal_weight * ce
    
    return K.mean(focal)


# Loss function registry
LOSS_FUNCTIONS = {
    "dice": dice_loss,
    "dice_bce": dice_bce_loss,
    "tversky": tversky_loss,
    "focal_tversky": focal_tversky_loss,
    "bce_tversky": bce_tversky_loss,
    "focal": focal_loss,
}


def get_loss_function(loss_name: str):
    """
    Get a loss function by name.
    
    Args:
        loss_name: Name of the loss function.
            Options: "dice", "dice_bce", "tversky", "focal_tversky", "bce_tversky", "focal"
            
    Returns:
        Loss function.
        
    Raises:
        ValueError: If loss_name is not recognized.
    """
    if loss_name not in LOSS_FUNCTIONS:
        raise ValueError(
            f"Unknown loss: {loss_name}. "
            f"Available losses: {list(LOSS_FUNCTIONS.keys())}"
        )
    return LOSS_FUNCTIONS[loss_name]
