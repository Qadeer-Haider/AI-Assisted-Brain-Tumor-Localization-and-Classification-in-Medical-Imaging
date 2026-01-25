"""
Training callbacks for classification.

Provides callback configurations for checkpointing, early stopping,
learning rate scheduling, and logging.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from tensorflow.keras.callbacks import (
    Callback,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

from ...utils.constants import DEFAULT_PATIENCE, LOGS_DIR, WEIGHTS_DIR


def get_callbacks(
    model_name: str,
    checkpoint_dir: Optional[Path] = None,
    log_dir: Optional[Path] = None,
    patience: int = DEFAULT_PATIENCE,
    reduce_lr_patience: int = 10,
    reduce_lr_factor: float = 0.5,
    monitor: str = "val_loss",
    mode: str = "min",
    use_tensorboard: bool = False,
) -> List[Callback]:
    """
    Create a list of training callbacks.
    
    Includes:
    - EarlyStopping: Stop training if no improvement
    - ModelCheckpoint: Save best model weights
    - CSVLogger: Log metrics to CSV file
    - ReduceLROnPlateau: Reduce learning rate when stuck
    - TensorBoard: Optional visualization
    
    Args:
        model_name: Name of the model (used for file naming).
        checkpoint_dir: Directory to save model weights.
        log_dir: Directory to save training logs.
        patience: Epochs to wait before early stopping.
        reduce_lr_patience: Epochs to wait before reducing LR.
        reduce_lr_factor: Factor to reduce learning rate by.
        monitor: Metric to monitor for callbacks.
        mode: 'min' for loss, 'max' for accuracy.
        use_tensorboard: Whether to enable TensorBoard logging.
        
    Returns:
        List of Keras callbacks.
        
    Example:
        >>> callbacks = get_callbacks("ConvNeXtBase", patience=30)
        >>> model.fit(train_ds, callbacks=callbacks)
    """
    if checkpoint_dir is None:
        checkpoint_dir = WEIGHTS_DIR / "classification"
    if log_dir is None:
        log_dir = LOGS_DIR / "classification"
    
    # Ensure base directories exist
    checkpoint_dir = Path(checkpoint_dir)
    log_dir = Path(log_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if weights already exist - if so, create timestamped folder
    base_checkpoint_path = checkpoint_dir / f"{model_name}_best_weights.keras"
    
    if base_checkpoint_path.exists():
        # Create timestamped run folder
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = f"{model_name}_{timestamp}"
        
        checkpoint_dir = checkpoint_dir / run_folder
        log_dir = log_dir / run_folder
        
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"⚠️ Weights already exist! Creating new run folder: {run_folder}")
    
    # File paths
    checkpoint_path = checkpoint_dir / f"{model_name}_best_weights.keras"
    log_path = log_dir / f"{model_name}_training_log.csv"

    
    callbacks = [
        # Stop training if validation loss doesn't improve
        EarlyStopping(
            monitor=monitor,
            mode=mode,
            verbose=1,
            patience=patience,
            restore_best_weights=True,
        ),
        
        # Save the best model based on validation performance
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=monitor,
            verbose=1,
            save_best_only=True,
            save_weights_only=False,  # Save full model
            mode=mode,
        ),
        
        # Log metrics to CSV file for later analysis
        CSVLogger(
            str(log_path),
            separator=",",
            append=False,
        ),
        
        # Reduce learning rate when training plateaus
        ReduceLROnPlateau(
            monitor=monitor,
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            verbose=1,
            mode=mode,
        ),
    ]
    
    # Optional TensorBoard logging
    if use_tensorboard:
        tensorboard_dir = log_dir / "tensorboard" / model_name
        callbacks.append(
            TensorBoard(
                log_dir=str(tensorboard_dir),
                histogram_freq=1,
                write_graph=True,
            )
        )
    
    return callbacks


def get_callbacks_from_config(config: Dict[str, Any], model_name: str) -> List[Callback]:
    """
    Create callbacks from a configuration dictionary.
    
    Args:
        config: Configuration dictionary.
        model_name: Name of the model.
        
    Returns:
        List of Keras callbacks.
    """
    es_config = config.get("early_stopping", {})
    lr_config = config.get("reduce_lr", {})
    
    return get_callbacks(
        model_name=model_name,
        patience=es_config.get("patience", DEFAULT_PATIENCE),
        reduce_lr_patience=lr_config.get("patience", 10),
        reduce_lr_factor=lr_config.get("factor", 0.5),
        monitor=es_config.get("monitor", "val_loss"),
        use_tensorboard=config.get("use_tensorboard", False),
    )


class PrintMetricsCallback(Callback):
    """
    Custom callback to print metrics in a formatted way after each epoch.
    """
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Format metrics nicely
        metrics_str = " | ".join([
            f"{k}: {v:.4f}" for k, v in logs.items()
        ])
        
        print(f"\n📊 Epoch {epoch + 1}: {metrics_str}")
