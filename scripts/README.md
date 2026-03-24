# 📜 Scripts

Executable Python scripts for training, evaluation, and inference.

## Available Scripts

### Classification

#### `train_classifier.py`
Train a classification model from the command line.

**Usage**:
```bash
# Train with default configuration (ConvNeXt)
python scripts/train_classifier.py

# Train specific model
python scripts/train_classifier.py --model ResNet152V2 --epochs 100

# Train with custom config
python scripts/train_classifier.py --config configs/classification_config.yaml
```

**Arguments**:
- `--model`: Model architecture (ConvNeXt, ResNet152V2, DenseNet201, EfficientNetV2S, VGG16)
- `--epochs`: Number of training epochs
- `--config`: Path to YAML configuration file
- `--weights`: Pre-trained weights ('imagenet' or path)
- `--batch-size`: Batch size for training

---

#### `evaluate_classifier.py`
Evaluate a trained classification model on the test set.

**Usage**:
```bash
# Evaluate trained model
python scripts/evaluate_classifier.py --weights weights/classification/ConvNeXtBase_best_weights.keras

# Save confusion matrix
python scripts/evaluate_classifier.py --weights model.keras --save-cm confusion_matrix.png
```

**Arguments**:
- `--weights`: Path to trained model weights (.keras file)
- `--model`: Model architecture name
- `--batch-size`: Batch size for evaluation
- `--save-cm`: Path to save confusion matrix
- `--no-plot`: Disable confusion matrix plotting

**Outputs**:
- Detailed metrics (accuracy, precision, recall, F1)
- Confusion matrix visualization
- Per-class performance breakdown

---

#### `predict_single.py`
Make predictions on a single image.

**Usage**:
```bash
# Predict single image
python scripts/predict_single.py \
    --image data/brisc2025/classification_task/test/glioma/sample.jpg \
    --weights weights/classification/ConvNeXtBase_best_weights.keras
```

**Arguments**:
- `--image`: Path to input image
- `--weights`: Path to trained model weights
- `--model`: Model architecture name

**Outputs**:
- Predicted class
- Confidence scores for all classes
- Visualization (optional)

---

### Segmentation

#### `train_segmentor.py`
Train a segmentation model from the command line.

**Usage**:
```bash
# Train with default configuration (UNet + BCE-Tversky)
python scripts/train_segmentor.py

# Train specific model and loss
python scripts/train_segmentor.py --model AttentionUNet --loss bce_tversky

# Train with custom config
python scripts/train_segmentor.py --config configs/segmentation_config.yaml

# Custom hyperparameters
python scripts/train_segmentor.py \
    --model UNet \
    --loss focal_tversky \
    --epochs 200 \
    --batch-size 32 \
    --learning-rate 0.0001
```

**Arguments**:
- `--model`: Model architecture (UNet, AttentionUNet, ResUNetPP, SwinUNet)
- `--loss`: Loss function (bce_tversky, dice, dice_bce, tversky, focal_tversky)
- `--backbone`: Encoder backbone (ResNet50, default)
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size for training
- `--learning-rate`: Learning rate
- `--img-size`: Input image size (default: 256)
- `--config`: Path to YAML configuration file

**Outputs**:
- Trained model saved to `weights/segmentation/`
- Training logs saved to `logs/segmentation/`
- CSV training history

---

#### `predict_mask.py`
Generate segmentation masks for images.

**Usage**:
```bash
# Predict single image with visualization
python scripts/predict_mask.py \
    --image data/brisc2025/segmentation_task/test/images/sample.jpg \
    --model weights/segmentation/UNet_bce_tversky_best.keras \
    --visualize

# Predict directory of images
python scripts/predict_mask.py \
    --input-dir data/brisc2025/segmentation_task/test/images \
    --model weights/segmentation/UNet_bce_tversky_best.keras \
    --output-dir outputs/predictions

# Save masks without visualization
python scripts/predict_mask.py \
    --image path/to/image.jpg \
    --model weights/segmentation/UNet_bce_tversky_best.keras \
    --output outputs/mask.png
```

**Arguments**:
- `--image`: Path to single input image
- `--input-dir`: Directory containing multiple images
- `--model`: Path to trained segmentation model
- `--output`: Output path for single mask
- `--output-dir`: Output directory for multiple masks
- `--visualize`: Show visualization with overlay
- `--threshold`: Binary threshold (default: 0.5)
- `--img-size`: Image size (default: 256)

**Outputs**:
- Predicted segmentation masks
- Overlay visualizations (if --visualize)
- Binary masks saved as PNG

---

### Deployment

#### `convert_models_to_onnx.py`
Convert all trained `.keras` classification and segmentation models to ONNX format for efficient deployment.

**Usage**:
```bash
# Automatically converts all models found in weights/{task} to weights/onnx/{task}
python scripts/convert_models_to_onnx.py
```

**Outputs**:
- Converted models saved to `weights/onnx/classification/`
- Converted models saved to `weights/onnx/segmentation/`

---

## Tips

- Use `--help` flag with any script to see all available options
- Scripts automatically create output directories if they don't exist
- Classification logs are saved to `logs/classification/`
- Segmentation logs are saved to `logs/segmentation/`
- Classification weights are saved to `weights/classification/`
- Segmentation weights are saved to `weights/segmentation/`
- All scripts support YAML configuration files for reproducibility

---

## Examples

### Classification Pipeline
```bash
# 1. Train model
python scripts/train_classifier.py --model ConvNeXtBase --epochs 100

# 2. Evaluate on test set
python scripts/evaluate_classifier.py \
    --weights weights/classification/ConvNeXtBase_best_weights.keras

# 3. Predict on new image
python scripts/predict_single.py \
    --image path/to/mri_scan.jpg \
    --weights weights/classification/ConvNeXtBase_best_weights.keras
```

### Segmentation Pipeline
```bash
# 1. Train segmentation model
python scripts/train_segmentor.py --model UNet --loss bce_tversky --epochs 200

# 2. Predict masks for test images
python scripts/predict_mask.py \
    --input-dir data/brisc2025/segmentation_task/test/images \
    --model weights/segmentation/UNet_bce_tversky_best.keras \
    --output-dir outputs/test_predictions \
    --visualize

# 3. Predict single image with overlay
python scripts/predict_mask.py \
    --image path/to/mri_scan.jpg \
    --model weights/segmentation/UNet_bce_tversky_best.keras \
    --visualize
```

### Batch Evaluation

**Classification:**
```bash
# Evaluate all classification models
for model in ConvNeXtBase ResNet152V2 DenseNet201 EfficientNetV2S VGG16; do
    python scripts/evaluate_classifier.py \
        --weights weights/classification/${model}_best_weights.keras \
        --model $model
done
```

**Segmentation:**
```bash
# Train multiple segmentation models with different architectures
for model in UNet AttentionUNet ResUNetPP; do
    python scripts/train_segmentor.py \
        --model $model \
        --loss bce_tversky \
        --epochs 200
done
```

### Experiment with Different Losses
```bash
# Train same model with different loss functions
for loss in dice dice_bce tversky focal_tversky bce_tversky; do
    python scripts/train_segmentor.py \
        --model UNet \
        --loss $loss \
        --epochs 200
done
```

---

## Quick Reference

| Task | Script | Primary Use |
|------|--------|-------------|
| Train classifier | `train_classifier.py` | Train CNN for tumor classification |
| Evaluate classifier | `evaluate_classifier.py` | Get metrics on test set |
| Predict class | `predict_single.py` | Single image classification |
| Train segmentor | `train_segmentor.py` | Train U-Net for tumor segmentation |
| Predict mask | `predict_mask.py` | Generate segmentation masks |
| Convert to ONNX | `convert_models_to_onnx.py` | Convert Keras models to ONNX |

---

## Configuration Files

All scripts can use YAML configuration files for consistent hyperparameters:

- **Classification**: `configs/classification_config.yaml`
- **Segmentation**: `configs/segmentation_config.yaml`

Example usage:
```bash
python scripts/train_classifier.py --config configs/classification_config.yaml
python scripts/train_segmentor.py --config configs/segmentation_config.yaml
```
