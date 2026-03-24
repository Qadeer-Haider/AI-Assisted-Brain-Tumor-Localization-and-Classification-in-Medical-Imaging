"""
Training callbacks for segmentation.

Provides callback factory functions for model checkpointing,
early stopping, learning rate scheduling, and logging.
"""

from pathlib import Path
from typing import List, Optional

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)


def get_callbacks(
    checkpoint_path: str,
    log_path: str,
    monitor_metric: str = "val_dice_coefficient",
    monitor_mode: str = "max",
    early_stopping_patience: int = 40,
    reduce_lr_patience: int = 10,
    reduce_lr_factor: float = 0.5,
    min_lr: float = 1e-7,
    use_tensorboard: bool = False,
    tensorboard_log_dir: Optional[str] = None,
) -> List[keras.callbacks.Callback]:
    """
    Create training callbacks for segmentation.
    
    Args:
        checkpoint_path: Path to save model checkpoints.
        log_path: Path for CSV training log.
        monitor_metric: Metric to monitor for checkpointing.
        monitor_mode: 'max' or 'min' for the monitored metric.
        early_stopping_patience: Epochs to wait before early stopping.
        reduce_lr_patience: Epochs to wait before reducing LR.
        reduce_lr_factor: Factor to reduce LR by.
        min_lr: Minimum learning rate.
        use_tensorboard: Whether to use TensorBoard logging.
        tensorboard_log_dir: Directory for TensorBoard logs.
        
    Returns:
        List of Keras callbacks.
    """
    callbacks = []
    
    # Ensure directories exist
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Early stopping on validation loss
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=early_stopping_patience,
        verbose=1,
        restore_best_weights=True,
    )
    callbacks.append(early_stopping)
    
    # Model checkpoint on best metric
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=monitor_metric,
        mode=monitor_mode,
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
    )
    callbacks.append(checkpoint)
    
    # CSV logger
    csv_logger = CSVLogger(
        log_path,
        separator=",",
        append=False,
    )
    callbacks.append(csv_logger)
    
    # Learning rate reduction
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        min_lr=min_lr,
        verbose=1,
    )
    callbacks.append(reduce_lr)
    
    # TensorBoard (optional)
    if use_tensorboard and tensorboard_log_dir:
        Path(tensorboard_log_dir).mkdir(parents=True, exist_ok=True)
        tensorboard = TensorBoard(
            log_dir=tensorboard_log_dir,
            histogram_freq=1,
            write_graph=True,
        )
        callbacks.append(tensorboard)
    
    return callbacks


def get_callbacks_from_config(config: dict, model_name: str, loss_name: str) -> List[keras.callbacks.Callback]:
    """
    Create callbacks from a configuration dictionary.
    
    Args:
        config: Configuration dictionary with callback settings.
        model_name: Name of the model (for naming files).
        loss_name: Name of the loss function (for naming files).
        
    Returns:
        List of Keras callbacks.
    """
    # Build paths
    weights_dir = config.get("weights_dir") or "weights/segmentation"
    logs_dir = config.get("logs_dir") or "logs/segmentation"
    
    checkpoint_path = f"{weights_dir}/{model_name}_{loss_name}_best.keras"
    log_path = f"{logs_dir}/{model_name}_{loss_name}_training_log.csv"
    
    # Early stopping config
    es_config = config.get("early_stopping", {})
    monitor_metric = es_config.get("monitor", "val_dice_coefficient")
    monitor_mode = es_config.get("mode", "max")
    es_patience = es_config.get("patience", 40)
    
    # Reduce LR config
    lr_config = config.get("reduce_lr", {})
    lr_patience = lr_config.get("patience", 10)
    lr_factor = lr_config.get("factor", 0.5)
    
    # TensorBoard
    use_tb = config.get("use_tensorboard", False)
    tb_dir = f"{logs_dir}/tensorboard/{model_name}_{loss_name}" if use_tb else None
    
    return get_callbacks(
        checkpoint_path=checkpoint_path,
        log_path=log_path,
        monitor_metric=monitor_metric,
        monitor_mode=monitor_mode,
        early_stopping_patience=es_patience,
        reduce_lr_patience=lr_patience,
        reduce_lr_factor=lr_factor,
        use_tensorboard=use_tb,
        tensorboard_log_dir=tb_dir,
    )
