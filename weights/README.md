# 💾 Model Weights

Directory for storing trained model weights.

## Structure

```
weights/
├── classification/          # Classification model weights
│   ├── ConvNeXtBase_best_weights.keras
│   ├── ResNet152V2_best_weights.keras
│   ├── DenseNet201_best_weights.keras
│   ├── EfficientNetV2S_best_weights.keras
│   └── VGG16_best_weights.keras
└── segmentation/           # Segmentation model weights
    ├── UNet_bce_tversky_best.keras
    ├── AttentionUNet_bce_tversky_best.keras
    ├── ResUNetPP_bce_tversky_best.keras
    └── SwinUNet_bce_tversky_best.keras
```

---

## Trained Models

### Classification Models

| Model | File | Size | Test Accuracy |
|-------|------|------|---------------|
| **DenseNet201** | `DenseNet201_best_weights.keras` | ~79 MB | 96.60% |
| **ConvNeXt** | `ConvNeXtBase_best_weights.keras` | ~339 MB | 95.60% |
| **ResNet152V2** | `ResNet152V2_best_weights.keras` | ~231 MB | 95.50% |
| **EfficientNetV2S** | `EfficientNetV2S_best_weights.keras` | ~84 MB | 95.40% |
| **VGG16** | `VGG16_best_weights.keras` | ~59 MB | 95.20% |

> **Note**: Run `notebooks/evaluate_all_models.ipynb` to populate accuracy values

### Segmentation Models

| Model | File | Size | Test Dice Score |
|-------|------|------|----------------|
| **UNet** | `UNet_bce_tversky_best.keras` | ~90 MB | TBD |
| **AttentionUNet** | `AttentionUNet_bce_tversky_best.keras` | ~95 MB | TBD |
| **ResUNetPP** | `ResUNetPP_bce_tversky_best.keras` | ~100 MB | TBD |
| **SwinUNet** | `SwinUNet_bce_tversky_best.keras` | ~110 MB | TBD |

> **Note**: Run `notebooks/test_segmentation.ipynb` to populate Dice scores

---

## File Format

All weights are saved in Keras native format (`.keras`):
- Single file containing model architecture and weights
- Fast loading with `tf.keras.models.load_model()`
- Includes optimizer state for resuming training

---

## Loading Weights

### Python API
```python
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('weights/classification/ConvNeXtBase_best_weights.keras')

# Make predictions
predictions = model.predict(images)
```

### Using Classifier Class
```python
from src.classification.inference import TumorClassifier

classifier = TumorClassifier('weights/classification/ConvNeXtBase_best_weights.keras')
result = classifier.predict('path/to/image.jpg')
```

### Using Segmentor Class
```python
from src.segmentation.inference import TumorSegmentor

segmentor = TumorSegmentor('weights/segmentation/UNet_bce_tversky_best.keras')
mask = segmentor.predict('path/to/image.jpg')
segmentor.visualize('path/to/image.jpg', overlay=True)
```

### From Command Line

**Classification:**
```bash
python scripts/predict_single.py \
    --weights weights/classification/ConvNeXtBase_best_weights.keras \
    --image path/to/image.jpg
```

**Segmentation:**
```bash
python scripts/predict_mask.py \
    --model weights/segmentation/UNet_bce_tversky_best.keras \
    --image path/to/image.jpg \
    --visualize
```

---

## Training Your Own

To train new models:

**Classification:**
```bash
python scripts/train_classifier.py --model ConvNeXtBase --epochs 100
```

**Segmentation:**
```bash
python scripts/train_segmentor.py --model UNet --loss bce_tversky --epochs 200
```

Weights will be automatically saved with naming conventions:
- Classification: `weights/classification/{ModelName}_best_weights.keras`
- Segmentation: `weights/segmentation/{ModelName}_{LossName}_best.keras`
- Best weights based on validation metrics
- Checkpoints saved during training

---

## Git LFS

Due to large file sizes, model weights should be tracked with Git LFS:

```bash
# Install Git LFS
git lfs install

# Track .keras files
git lfs track "*.keras"

# Add and commit
git add .gitattributes
git add weights/
git commit -m "Add trained model weights"
```

---

## Download Pre-trained Weights

If weights are too large for GitHub, download from:
- **Google Drive**: [Download Weights](https://drive.google.com/drive/folders/1xtkmtTv6SF95hfEbGJkB1J7j5VA6-P8k?usp=drive_link)

Extract to `weights/classification/` or `weights/segmentation/` directory.

---

## Weight File Naming Convention

**Classification Format:** `{ModelName}_best_weights.keras`

Examples:
- ✅ `ConvNeXtBase_best_weights.keras`
- ✅ `ResNet152V2_best_weights.keras`

**Segmentation Format:** `{ModelName}_{LossName}_best.keras`

Examples:
- ✅ `UNet_bce_tversky_best.keras`
- ✅ `AttentionUNet_focal_tversky_best.keras`

**Avoid:**
- ❌ `model.keras` (not descriptive)
- ❌ `best.h5` (wrong format)

---

## Storage Recommendations

- Keep only best weights for each model (not all checkpoints)
- Use Git LFS for version control
- Compress old models if disk space is limited
- Document model performance in README or commit message

---

## Troubleshooting

### "File not found" Error
Ensure weights exist in correct directory:
```bash
ls weights/classification/
ls weights/segmentation/
```

### "Model loading failed"
Check TensorFlow version compatibility:
```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

Weights require TensorFlow 2.10+

### Large File Warning
Use Git LFS to avoid repository bloat:
```bash
git lfs migrate import --include="*.keras"
```
