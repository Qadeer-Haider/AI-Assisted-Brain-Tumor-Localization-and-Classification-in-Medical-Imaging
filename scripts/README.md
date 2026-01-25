# 📜 Scripts

Executable Python scripts for training, evaluation, and inference.

## Available Scripts

### Training

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

### Evaluation

#### `evaluate_classifier.py`
Evaluate a trained model on the test set.

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

### Inference

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

## Segmentation Scripts (Coming Soon)

#### `train_segmentor.py`
Train a segmentation model (placeholder).

#### `predict_mask.py`
Generate segmentation masks (placeholder).

---

## Tips

- Use `--help` flag with any script to see all available options
- Scripts automatically create output directories if they don't exist
- Training logs are saved to `logs/classification/`
- Model weights are saved to `weights/classification/`
- All scripts support YAML configuration files for reproducibility

## Examples

### Training Pipeline
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

### Batch Evaluation
```bash
# Evaluate all models
for model in ConvNeXtBase ResNet152V2 DenseNet201 EfficientNetV2S VGG16; do
    python scripts/evaluate_classifier.py \
        --weights weights/classification/${model}_best_weights.keras \
        --model $model
done
```
