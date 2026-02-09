# 📦 Source Code

Core implementation of the brain tumor classification and segmentation framework.

## Directory Structure

```
src/
├── classification/          # Classification module
│   ├── data/               # Data loading & preprocessing
│   ├── models/             # Model architectures
│   ├── training/           # Training utilities
│   └── inference/          # Prediction & evaluation
├── segmentation/           # Segmentation module
│   ├── data/               # Data loading & preprocessing
│   ├── models/             # U-Net architectures
│   ├── training/           # Training utilities
│   └── inference/          # Prediction & evaluation
└── utils/                  # Shared utilities
    ├── config.py          # Configuration management
    ├── constants.py       # Global constants
    └── visualization.py   # Plotting utilities
```

---

## Modules

### 📊 Classification (`src/classification/`)

Complete implementation of brain tumor classification.

#### **Data** (`classification/data/`)
- `utils.py` - DataFrame building, stratified splitting, class weights
- `dataset.py` - TensorFlow dataset creation with preprocessing
- `preprocessing.py` - Model-specific preprocessing functions

**Key Functions**:
```python
from src.classification.data import build_dataframe, make_dataset

# Build DataFrame from images
df = build_dataframe(split="train")

# Create TensorFlow dataset
train_ds = make_dataset(df, model_name="ConvNeXtBase", batch_size=32)
```

#### **Models** (`classification/models/`)
- `backbones.py` - Pre-trained backbone loaders (ResNet, DenseNet, etc.)
- `heads.py` - Classification heads and attention mechanisms
- `builder.py` - Model assembly (backbone + head)

**Key Functions**:
```python
from src.classification.models import build_model

# Build complete model
model = build_model(
    model_name="ConvNeXtBase",
    num_classes=4,
    dropout_rate=0.3,
    dense_units=256
)
```

#### **Training** (`classification/training/`)
- `trainer.py` - High-level training orchestration
- `callbacks.py` - Custom Keras callbacks
- `metrics.py` - Medical-focused metrics (recall-oriented)

**Key Classes**:
```python
from src.classification.training import ClassificationTrainer

trainer = ClassificationTrainer(model_name="ConvNeXtBase")
trainer.prepare_data()
trainer.build()
trainer.compile()
trainer.train(epochs=100)
```

#### **Inference** (`classification/inference/`)
- `predictor.py` - Single image prediction
- `evaluator.py` - Model evaluation with metrics

**Key Classes**:
```python
from src.classification.inference import TumorClassifier, ClassificationEvaluator

# Prediction
classifier = TumorClassifier("path/to/model.keras")
result = classifier.predict("path/to/image.jpg")

# Evaluation
evaluator = ClassificationEvaluator(model, test_ds)
metrics = evaluator.evaluate()
evaluator.print_classification_report()
```

---

### 🎯 Segmentation (`src/segmentation/`)

Complete implementation of brain tumor segmentation.

#### **Data** (`segmentation/data/`)
- `dataset.py` - Image-mask pair loading with stratified splitting
- `preprocessing.py` - Albumentations augmentation pipeline

**Key Functions**:
```python
from src.segmentation.data import create_segmentation_datasets

# Create train/val/test datasets
train_ds, val_ds, test_ds = create_segmentation_datasets(
    train_images_dir="data/train/images",
    train_masks_dir="data/train/masks",
    img_size=(256, 256),
    batch_size=32,
    use_augmentation=True
)
```

#### **Models** (`segmentation/models/`)
- `architectures.py` - U-Net variants (UNet, AttentionUNet, ResUNetPP, SwinUNet)
- `builder.py` - Model factory with backbone support

**Key Functions**:
```python
from src.segmentation.models import build_segmentation_model

# Build segmentation model
model = build_segmentation_model(
    model_name="UNet",
    input_shape=(256, 256, 3),
    num_classes=1,
    backbone="ResNet50",
    freeze_backbone=True
)
```

#### **Training** (`segmentation/training/`)
- `trainer.py` - High-level training orchestration
- `losses.py` - Segmentation losses (Dice, Tversky, BCE-Tversky, Focal)
- `metrics.py` - Segmentation metrics (Dice, IoU, Sensitivity, Specificity)
- `callbacks.py` - Training callbacks

**Key Classes**:
```python
from src.segmentation.training import SegmentationTrainer

trainer = SegmentationTrainer(
    model_name="UNet",
    loss_name="bce_tversky"
)
trainer.prepare_data("data/train/images", "data/train/masks")
trainer.build()
trainer.compile()
trainer.train(epochs=200)
```

#### **Inference** (`segmentation/inference/`)
- `predictor.py` - Mask prediction and visualization

**Key Classes**:
```python
from src.segmentation.inference import TumorSegmentor

# Prediction
segmentor = TumorSegmentor("path/to/model.keras")
mask = segmentor.predict("path/to/image.jpg")

# Visualization
segmentor.visualize("path/to/image.jpg", overlay=True)
```

---

### 🛠️ Utils (`src/utils/`)

Shared utilities across modules.

#### `config.py`
Configuration file loading and management.

```python
from src.utils.config import load_config

config = load_config("configs/classification_config.yaml")
```

#### `constants.py`
Global constants and configurations.

```python
from src.utils.constants import CLASS_NAMES, NUM_CLASSES, INPUT_SHAPE

CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
NUM_CLASSES = 4
INPUT_SHAPE = (224, 224, 3)
```

#### `visualization.py`
Plotting and visualization utilities.

```python
from src.utils.visualization import plot_confusion_matrix, plot_training_history

plot_confusion_matrix(cm, class_names=CLASS_NAMES)
```

---

## Design Principles

### 1. **Modularity**
Each component has a single responsibility and clear interfaces.

### 2. **Reusability**
Common functionality is abstracted into utilities.

### 3. **Extensibility**
Easy to add new models, metrics, or preprocessing methods.

### 4. **Type Hints**
All functions include type annotations for better IDE support.

### 5. **Documentation**
Comprehensive docstrings with examples.

---

## Adding New Features

### Add a New Model
1. Add backbone to `classification/models/backbones.py`
2. Register in `AVAILABLE_MODELS` in `utils/constants.py`
3. (Optional) Add custom preprocessing in `classification/data/preprocessing.py`

### Add a New Metric
1. Implement in `classification/training/metrics.py`
2. Add to compile step in `trainer.py`

### Modify Classification Head
Edit `add_universal_head()` in `classification/models/heads.py`

---

## Import Conventions

```python
# Classification Data
from src.classification.data import build_dataframe, make_dataset

# Classification Models
from src.classification.models import build_model

# Classification Training
from src.classification.training import ClassificationTrainer

# Classification Inference
from src.classification.inference import TumorClassifier, ClassificationEvaluator

# Segmentation Data
from src.segmentation.data import create_segmentation_datasets, get_training_augmentation

# Segmentation Models
from src.segmentation.models import build_segmentation_model, AVAILABLE_SEGMENTATION_MODELS

# Segmentation Training
from src.segmentation.training import SegmentationTrainer, get_loss_function, get_segmentation_metrics

# Segmentation Inference
from src.segmentation.inference import TumorSegmentor, predict_mask

# Utils
from src.utils.config import load_config
from src.utils.constants import CLASS_NAMES, NUM_CLASSES
from src.utils.visualization import plot_confusion_matrix
```

---

## Testing

Run tests with:
```bash
pytest tests/
```

Ensure all changes pass type checking:
```bash
mypy src/
```
