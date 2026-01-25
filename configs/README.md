# ⚙️ Configuration Files

YAML configuration files for training and evaluation.

## Available Configurations

### `classification_config.yaml`
Configuration for brain tumor classification training.

**Key Settings**:

#### Model Settings
```yaml
model_name: "ConvNeXtBase"          # Architecture to use
use_attention: false                 # SE attention blocks (optional)
trainable_backbone: false            # Fine-tune backbone weights
dropout_rate: 0.3                    # Dropout for regularization
dense_units: 256                     # Units in classification head
```

**Available Models**:
- `ConvNeXtBase` - State-of-the-art performance (~88M params)
- `ResNet152V2` - Deep feature extraction (~58M params)
- `DenseNet201` - Feature reuse & efficiency (~18M params)
- `EfficientNetV2S` - Balance of speed & accuracy (~21M params)
- `VGG16` - Classic architecture (~14M params)

#### Data Settings
```yaml
img_size: [224, 224]                # Input image dimensions
batch_size: 32                       # Training batch size
val_split: 0.2                       # Validation split ratio
random_seed: 42                      # Reproducibility seed
```

#### Training Settings
```yaml
learning_rate: 0.0001               # Initial learning rate
epochs: 300                          # Maximum training epochs
use_class_weights: true              # Balance class distribution
```

#### Augmentation Settings
```yaml
augmentation:
  enabled: true                      # Enable data augmentation
  horizontal_flip: true              # Random horizontal flips
  rotation_range: 0.028              # ~10 degrees rotation
```

#### Callback Settings
```yaml
early_stopping:
  monitor: "val_loss"                # Metric to monitor
  patience: 30                       # Epochs without improvement
  restore_best_weights: true         # Restore best weights

reduce_lr:
  monitor: "val_loss"                # Metric to monitor
  factor: 0.5                        # LR reduction factor
  patience: 10                       # Epochs before reduction
```

---

### `segmentation_config.yaml`
Configuration for tumor segmentation (coming soon).

---

## Usage

### From Scripts
```bash
# Use config file directly
python scripts/train_classifier.py --config configs/classification_config.yaml
```

### From Python
```python
from src.utils.config import load_config

config = load_config("configs/classification_config.yaml")
model_name = config["model_name"]
epochs = config["epochs"]
```

---

## Customization Tips

1. **Experiment with different models**: Change `model_name` to try different architectures
2. **Adjust batch size**: Reduce if you run out of memory
3. **Tune learning rate**: Lower values (1e-5) for fine-tuning
4. **Enable/disable augmentation**: Based on dataset size
5. **Modify early stopping**: Increase patience for longer training

---

## Configuration Hierarchy

Settings can be overridden in this order (highest priority first):
1. Command-line arguments
2. YAML configuration file
3. Default values in code

Example:
```bash
# Override epochs from config
python scripts/train_classifier.py \
    --config configs/classification_config.yaml \
    --epochs 50  # This overrides config value
```

---

## Best Practices

- ✅ Use YAML configs for reproducible experiments
- ✅ Keep separate configs for different experiments
- ✅ Version control your config files
- ✅ Document any non-standard settings
- ✅ Start with default values and tune incrementally

---

## Medical Imaging Considerations

For medical imaging tasks:
- Conservative augmentation (small rotations only)
- Class weighting for imbalanced datasets
- Focus on recall to minimize false negatives
- Multiple validation metrics (not just accuracy)
