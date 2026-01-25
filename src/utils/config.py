"""
Configuration management utilities.

Provides YAML configuration loading and validation for training runs.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml

from .constants import (
    CONFIGS_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_PATIENCE,
    DEFAULT_VAL_SPLIT,
    IMG_SIZE,
    RANDOM_SEED,
)


def load_config(config_path: Optional[str] = None, task: str = "classification") -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML config file. If None, loads default config.
        task: Task type ('classification' or 'segmentation') for default config.
        
    Returns:
        Dictionary containing configuration parameters.
        
    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the config file is invalid YAML.
    """
    if config_path is None:
        config_path = CONFIGS_DIR / f"{task}_config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        print(f"⚠️ Config file not found: {config_path}")
        print("📋 Using default configuration...")
        return get_default_config(task)
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Merge with defaults for any missing keys
    defaults = get_default_config(task)
    merged = {**defaults, **config}
    
    return merged


def get_default_config(task: str = "classification") -> Dict[str, Any]:
    """
    Get default configuration for training.
    
    Args:
        task: Task type ('classification' or 'segmentation').
        
    Returns:
        Dictionary with default configuration values.
    """
    base_config = {
        # Data
        "img_size": list(IMG_SIZE),
        "batch_size": DEFAULT_BATCH_SIZE,
        "val_split": DEFAULT_VAL_SPLIT,
        "random_seed": RANDOM_SEED,
        
        # Training
        "learning_rate": DEFAULT_LEARNING_RATE,
        "epochs": DEFAULT_EPOCHS,
        "patience": DEFAULT_PATIENCE,
        
        # Callbacks
        "early_stopping": {
            "monitor": "val_loss",
            "patience": DEFAULT_PATIENCE,
            "restore_best_weights": True,
        },
        "reduce_lr": {
            "monitor": "val_loss",
            "factor": 0.5,
            "patience": 10,
        },
        
        # Augmentation
        "augmentation": {
            "horizontal_flip": True,
            "rotation_range": 0.028,
        },
    }
    
    if task == "classification":
        base_config["model_name"] = "ConvNeXtBase"
        base_config["use_class_weights"] = True
    elif task == "segmentation":
        base_config["model_name"] = "UNet"
        base_config["loss"] = "dice"
    
    return base_config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary to save.
        save_path: Path to save the config file.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Config saved to: {save_path}")
