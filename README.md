# 🧠 AI-Assisted Brain Tumor Localization and Classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange?logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)
![Phase](https://img.shields.io/badge/Phase-1_Classification-blueviolet)

**A deep learning framework for brain tumor detection and localization using MRI images**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Models](#-models) • [Dataset](#-dataset) • [Results](#-results)

</div>

---

## 📋 Overview

This project is a comprehensive deep learning solution for medical imaging analysis, designed to assist in the automated detection of brain tumors.

**Current Focus (Phase 1): Classification**
- Robustly classifying MRI slices into 4 categories: **Glioma, Meningioma, Pituitary, and No Tumor**.
- Utilizing advanced transfer learning architectures (ConvNeXt, EfficientNetV2, etc.).
- Implementing rigorous evaluation metrics suitable for medical diagnosis support.

**Future Focus (Phase 2): Segmentation** *(Coming Soon)*
- Pixel-wise localization of tumor regions to assist in surgical planning.

## ✨ Features

### ✅ Classification (Current)
- 🏗️ **Multiple Architectures**: ResNet, DenseNet, VGG, EfficientNetV2, ConvNeXt
- 🔧 **Modular Design**: Production-ready codebase offering easy extensibility
- ⚙️ **YAML Configuration**: Centralized hyperparameter management
- 📊 **Medical Metrics**: Optimized for Recall and F1-Score to minimize false negatives
- 📈 **Class Balancing**: Automatic weight computation for imbalanced datasets
- 🎨 **Visualizations**: Comprehensive performance plots (AUC, Confusion Matrices)

### 🚧 Segmentation (Upcoming)
- 🧩 **U-Net Integration**: Implementation of U-Net and Attention U-Net
- 🎯 **Mask Generation**: Precise tumor boundary detection
- 📐 **Dice Coefficient**: Advanced segmentation evaluation metrics

## 📁 Project Structure

```
AI-Assisted-Brain-Tumor-Localization-and-Classification/
├── configs/                    # YAML configuration files
│   ├── classification_config.yaml
│   └── segmentation_config.yaml
├── data/                       # Dataset directory
│   └── brisc2025/
│       ├── classification_task/ # Current active dataset
│       └── segmentation_task/
├── notebooks/                  # Jupyter notebooks
│   ├── evaluate_all_models.ipynb
│   ├── train_classification.ipynb
│   └── test_classification.ipynb
├── scripts/                    # Executable training/inference scripts
│   ├── train_classifier.py
│   ├── evaluate_classifier.py
│   ├── predict_single.py
│   ├── train_segmentor.py      # (Pending)
│   └── predict_mask.py         # (Pending)
├── src/                        # Main source package
│   ├── classification/         # Classification module (Complete)
│   ├── segmentation/           # Segmentation module (In Progress)
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

## 💻 Usage (Classification)

### Training
```bash
# Train with default configuration (ConvNeXt)
python scripts/train_classifier.py

# Train specific model
python scripts/train_classifier.py --model ResNet152V2 --epochs 100
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

## 🏗️ Classification Models

We leverage state-of-the-art architectures initialized with ImageNet weights and fine-tuned for medical imaging:

| Model | Parameters | Best Use Case |
|-------|------------|---------------|
| **ConvNeXt** | ~88M | State-of-the-art performance |
| **EfficientNetV2** | ~21M | Balance of speed & accuracy |
| **ResNet152V2** | ~58M | Deep feature extraction |
| **DenseNet201** | ~18M | Feature reuse & efficiency |
| **VGG16** | ~14M | Classic baseline |

## 📊 Dataset (Classification Task)

This project uses the **BRISC2025** (Brain Tumor Image Segmentation & Classification) dataset.
**Note:** We are currently utilizing the **Classification subset** of this dataset.

- **6,000** T1-weighted MRI slices (5,000 train / 1,000 test)
- **4 classes**: Glioma, Meningioma, Pituitary Tumor, No Tumor
- **3 anatomical planes**: Axial, Coronal, Sagittal

### 🔄 Data Splitting Strategy (Classification)

To ensure robust evaluation for our **classification models** and prevent data leakage, we utilize a **Stratified Shuffle Split** technique:

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



### 🔍 Performance Visualization

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


## 📈 Results

Test set performance on 1,000 T1-weighted MRI slices from the BRISC2025 dataset:

| Model | Accuracy | Precision | Recall | F1 Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| **DenseNet201** | 96.60% | 96.71% | 96.94% | 96.81% | 99.45% |
| **ConvNeXt** | 95.60% | 95.82% | 95.97% | 95.85% | 99.63% |
| **ResNet152V2** | 95.50% | 95.59% | 96.00% | 95.78% | 99.64% |
| **EfficientNetV2S** | 95.40% | 95.89% | 95.80% | 95.81% | 99.57% |
| **VGG16** | 95.20% | 95.51% | 95.67% | 95.56% | 98.99% |

**Best Performing Model:** DenseNet201 with 96.60% accuracy

*All models use transfer learning with frozen backbones and custom classification heads.*
## 🗺️ Roadmap

- [x] **Phase 1: Classification**
    - [x] Data Loading & Preprocessing
    - [x] Model Implementation (ConvNeXt, EfficientNet, etc.)
    - [x] Training Pipeline & Logging
    - [x] Evaluation & Visualization
- [ ] **Phase 2: Segmentation**
    - [ ] U-Net Architecture Implementation
    - [ ] Mask Data Processing
    - [ ] Segmentation Training Loop
    - [ ] Mask Prediction Visualization
- [ ] **Phase 3: Deployment**
    - [ ] Web Interface (Streamlit)
    - [ ] Docker Containerization
    - [ ] ONNX Export

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
