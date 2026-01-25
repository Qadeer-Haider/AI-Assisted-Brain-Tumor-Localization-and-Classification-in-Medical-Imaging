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
└── segmentation/           # Segmentation model weights (coming soon)
```

---

## Trained Models

### Classification Models

| Model | File | Size | Test Accuracy |
|-------|------|------|---------------|
| **ConvNeXt Base** | `ConvNeXtBase_best_weights.keras` | ~339 MB | TBD |
| **ResNet152V2** | `ResNet152V2_best_weights.keras` | ~231 MB | TBD |
| **DenseNet201** | `DenseNet201_best_weights.keras` | ~79 MB | TBD |
| **EfficientNetV2S** | `EfficientNetV2S_best_weights.keras` | ~84 MB | TBD |
| **VGG16** | `VGG16_best_weights.keras` | ~59 MB | TBD |

> **Note**: Run `notebooks/evaluate_all_models.ipynb` to populate accuracy values

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

### From Command Line
```bash
python scripts/predict_single.py \
    --weights weights/classification/ConvNeXtBase_best_weights.keras \
    --image path/to/image.jpg
```

---

## Training Your Own

To train new models:

```bash
python scripts/train_classifier.py --model ConvNeXtBase --epochs 100
```

Weights will be automatically saved to `weights/classification/` with naming convention:
- `{ModelName}_best_weights.keras` - Best weights based on validation loss
- `{ModelName}_{timestamp}/` - Checkpoints during training

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
- **Google Drive**: [Link]
- **Hugging Face**: [Link]
- **Releases**: Check GitHub releases

Extract to `weights/classification/` directory.

---

## Weight File Naming Convention

Format: `{ModelName}_best_weights.keras`

Examples:
- ✅ `ConvNeXtBase_best_weights.keras`
- ✅ `ResNet152V2_best_weights.keras`
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
