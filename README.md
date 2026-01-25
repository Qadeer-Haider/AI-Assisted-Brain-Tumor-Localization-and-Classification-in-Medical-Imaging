# 🧠 AI-Assisted Brain Tumor Localization and Classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange?logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

**A deep learning framework for brain tumor detection and localization using MRI images**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Models](#-models) • [Dataset](#-dataset) • [Results](#-results)

</div>

---

## 📋 Overview

This project provides a modular deep learning framework for:

1. **Classification**: Detect and classify brain tumors into 4 categories (Glioma, Meningioma, Pituitary, No Tumor)
2. **Segmentation**: Localize tumors by predicting pixel-wise masks *(coming soon)*

Built with transfer learning using state-of-the-art architectures and trained on the BRISC2025 dataset.

## ✨ Features

- 🏗️ **Multiple Architectures**: ResNet, DenseNet, VGG, EfficientNetV2, ConvNeXt
- 🔧 **Modular Design**: Clean, well-organized codebase suitable for research and production
- ⚙️ **YAML Configuration**: Easy hyperparameter management without code changes
- 📊 **Medical Metrics**: Focus on recall to minimize false negatives (missing tumors)
- 📈 **Class Balancing**: Automatic class weight computation for imbalanced datasets
- 🎨 **Rich Output**: Beautiful console output with progress bars and formatted tables

## 📁 Project Structure

```
AI-Assisted-Brain-Tumor-Localization-and-Classification/
├── configs/                    # YAML configuration files
│   ├── classification_config.yaml
│   └── segmentation_config.yaml
├── data/                       # Dataset directory
│   └── brisc2025/
│       ├── classification_task/
│       └── segmentation_task/
├── notebooks/                  # Jupyter notebooks
│   ├── evaluate_all_models.ipynb
│   ├── train_classification.ipynb
│   └── test_classification.ipynb
├── scripts/                    # Executable training/inference scripts
│   ├── train_classifier.py
│   ├── evaluate_classifier.py
│   ├── predict_single.py
│   ├── train_segmentor.py      # Placeholder
│   └── predict_mask.py         # Placeholder
├── src/                        # Main source package
│   ├── classification/         # Classification module
│   │   ├── data/               # Data loading & preprocessing
│   │   ├── models/             # Model architectures
│   │   ├── training/           # Training utilities
│   │   └── inference/          # Prediction & evaluation
│   ├── segmentation/           # Segmentation module (placeholder)
│   │   ├── data/
│   │   ├── models/
│   │   ├── training/
│   │   └── inference/
│   └── utils/                  # Shared utilities
├── weights/                    # Saved model weights
├── logs/                       # Training logs
├── setup.py                    # Package installation
├── requirements.txt            # Dependencies
└── README.md
```


## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)

### Setup

```bash
# Clone the repository
git clone https://github.com/Qadeer-Haider/AI-Assisted-Brain-Tumor-Localization-and-Classification-in-Medical-Imaging.git
cd AI-Assisted-Brain-Tumor-Localization-and-Classification-in-Medical-Imaging

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## 💻 Usage

### Training

```bash
# Train with default configuration (ConvNeXt)
python scripts/train_classifier.py

# Train specific model
python scripts/train_classifier.py --model ResNet152V2 --epochs 100

# Train with custom config
python scripts/train_classifier.py --config configs/classification_config.yaml
```

### Evaluation

```bash
# Evaluate trained model
python scripts/evaluate_classifier.py --weights weights/classification/ConvNeXt_best_weights.keras
```

### Prediction

```bash
# Predict single image
python scripts/predict_single.py \
    --image data/brisc2025/classification_task/test/glioma/sample.jpg \
    --weights weights/classification/ConvNeXt_best_weights.keras
```

### Python API

```python
from src.classification import TumorClassifier, ClassificationTrainer

# Quick prediction
classifier = TumorClassifier("weights/classification/model.keras")
result = classifier.predict("path/to/mri_scan.jpg")
print(f"Prediction: {result['class']} ({result['confidence']:.1f}%)")

# Training
trainer = ClassificationTrainer(model_name="ConvNeXt")
trainer.prepare_data()
trainer.build()
trainer.compile()
trainer.train(epochs=100)
trainer.evaluate()
```

## 🏗️ Models

### Available Architectures

| Model | Parameters | Best Use Case |
|-------|------------|---------------|
| **ConvNeXt** | ~88M | State-of-the-art performance |
| **EfficientNetV2** | ~21M | Balance of speed & accuracy |
| **ResNet152V2** | ~58M | Deep feature extraction |
| **DenseNet201** | ~18M | Feature reuse & efficiency |
| **VGG16** | ~14M | Classic architecture |

### Architecture Overview

```
┌────────────────────────────────────────┐
│            Input Image                  │
│           (224 × 224 × 3)               │
└────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────┐
│         Backbone (Frozen)               │
│  (ResNet/DenseNet/EfficientNet/ConvNeXt)│
└────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────┐
│     Universal Classification Head       │
│  GAP → BN → Dense → Dropout (×2)       │
└────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────┐
│           Softmax Output                │
│    [glioma, meningioma, no_tumor,      │
│              pituitary]                 │
└────────────────────────────────────────┘
```

## 📊 Dataset

This project uses the **BRISC2025** (Brain Tumor Image Segmentation & Classification) dataset:

- **6,000** T1-weighted MRI slices (5,000 train / 1,000 test)
- **4 classes**: Glioma, Meningioma, Pituitary Tumor, No Tumor
- **3 anatomical planes**: Axial, Coronal, Sagittal
- Physician-validated pixel-level segmentation masks

### 🔄 Data Splitting Strategy

To ensure robust evaluation and prevent data leakage, we utilize a **Stratified Shuffle Split** technique:

- **Stratification Method**: Data is stratified based on *both* **Tumor Class** and **Anatomical Plane** (Axial, Coronal, Sagittal). This ensures that every subset of data preserves the original distribution of tumor types and viewing angles, preventing biases toward specific orientations.
- **Split Ratios**:
  - **Train**: 80% of the training data (used for model weight optimization)
  - **Validation**: 20% of the training data (used for hyperparameter tuning and early stopping)
  - **Test**: Separate hold-out set (~1,000 images) used strictly for final performance evaluation

### Citation

```bibtex
@article{fateh2025brisc,
  title={Brisc: Annotated dataset for brain tumor segmentation and classification with swin-hafnet},
  author={Fateh, Amirreza and Rezvani, Yasin and Moayedi, Sara and others},
  journal={arXiv preprint arXiv:2506.14318},
  year={2025}
}
```

## 📈 Results

> ℹ️ **Note**: The results and visualizations below are specific to the **Classification Task**.
>
> 🚧 **Segmentation Task**: Results and visualizations for tumor segmentation will be added soon!

### 🔍 Classification Performance

Here are the comparative results of our trained models. **ConvNeXtBase** and **EfficientNetV2S** demonstrated superior performance across key metrics.


#### 🏆 Model Comparison
![Model Comparison](logs/classification/all_models_comparison.png)

#### 📉 Training Dynamics
<div align="center">
  <img src="logs/classification/accuracy_curves.png" width="45%" />
  <img src="logs/classification/loss_curves.png" width="45%" />
</div>

#### 🎯 Metric Analysis (Recall & AUC)
<div align="center">
  <img src="logs/classification/recall_curves.png" width="45%" />
  <img src="logs/classification/auc_curves.png" width="45%" />
</div>

### Detailed Metrics

| Model | Accuracy | Recall | Precision | AUC |
|-------|----------|--------|-----------|-----|
| ConvNeXt | - | - | - | - |
| ResNet152V2 | - | - | - | - |
| EfficientNetV2 | - | - | - | - |



## 🗺️ Roadmap

- [x] Classification module
- [x] Multiple backbone support
- [x] Configuration management
- [x] Training scripts
- [ ] Segmentation module (U-Net)
- [ ] Web demo interface
- [ ] ONNX export for deployment
- [ ] Docker containerization

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- BRISC2025 dataset creators and annotators
- TensorFlow team for pre-trained models
- Medical imaging research community

---

<div align="center">

**Built with ❤️ by [Qadeer Haider](https://github.com/Qadeer-Haider)**

</div>
