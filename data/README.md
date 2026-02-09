# 📂 Data Directory

This directory contains the BRISC2025 dataset for brain tumor classification and segmentation.

## Structure

```
data/
├── brisc2025/                  # BRISC2025 Dataset
│   ├── classification_task/    # Classification dataset
│   │   ├── train/             # Training images (5,000 images)
│   │   │   ├── glioma/
│   │   │   ├── meningioma/
│   │   │   ├── no_tumor/
│   │   │   └── pituitary/
│   │   └── test/              # Test images (1,000 images)
│   │       ├── glioma/
│   │       ├── meningioma/
│   │       ├── no_tumor/
│   │       └── pituitary/
│   └── segmentation_task/     # Segmentation dataset
│       ├── train/
│       │   ├── images/        # Training MRI images
│       │   └── masks/         # Corresponding masks
│       └── test/
│           ├── images/        # Test MRI images
│           └── masks/         # Corresponding masks (if available)
├── raw/                       # Original untouched datasets (optional)
├── processed/                 # Preprocessed data (optional)
└── README.md                  # This file
```

---

## Dataset: BRISC2025

**Brain Tumor Image Segmentation & Classification Dataset**

This project uses the BRISC2025 dataset which contains T1-weighted MRI scans for both classification and segmentation tasks.

### Citation

```bibtex
@article{fateh2025brisc,
  title={Brisc: Annotated dataset for brain tumor segmentation and classification with swin-hafnet},
  author={Fateh, Amirreza and Rezvani, Yasin and Moayedi, Sara and others},
  journal={arXiv preprint arXiv:2506.14318},
  year={2025}
}
```

---

## Classification Task

### Dataset Details
- **Total Images**: 6,000 T1-weighted MRI slices
- **Train/Test Split**: 5,000 / 1,000
- **Classes**: 4 (Glioma, Meningioma, No Tumor, Pituitary)
- **Image Format**: RGB images
- **Anatomical Planes**: Axial, Coronal, Sagittal

### Class Distribution
The dataset is relatively balanced across classes with stratification by both tumor type and anatomical plane.

### Data Organization
```
classification_task/
├── train/
│   ├── glioma/         # Glioma tumor images
│   ├── meningioma/     # Meningioma tumor images
│   ├── no_tumor/       # Healthy brain scans
│   └── pituitary/      # Pituitary tumor images
└── test/
    └── [same structure as train]
```

---

## Segmentation Task

### Dataset Details
- **Task**: Pixel-wise tumor localization
- **Format**: Image-mask pairs
- **Mask Type**: Binary masks (tumor region = white, background = black)
- **Image Size**: Variable (resized during training)

### Data Organization
```
segmentation_task/
├── train/
│   ├── images/        # MRI scan images
│   └── masks/         # Binary segmentation masks
└── test/
    ├── images/        # Test MRI images
    └── masks/         # Ground truth masks 
```

### Preprocessing
- Images are automatically resized to `(256, 256)` during training
- Masks are binarized (0 = background, 1 = tumor)
- Data augmentation applied via Albumentations during training

---

## Data Loading

### Classification
```python
from src.classification.data import build_dataframe, create_datasets

# Build DataFrame from directory structure
df = build_dataframe(split="train")

# Create TensorFlow datasets
train_ds, val_ds, test_ds = create_datasets(
    train_df=df,
    model_name="ConvNeXtBase",
    batch_size=32
)
```

### Segmentation
```python
from src.segmentation.data import create_segmentation_datasets

# Create datasets from image/mask directories
train_ds, val_ds, test_ds = create_segmentation_datasets(
    train_images_dir="data/brisc2025/segmentation_task/train/images",
    train_masks_dir="data/brisc2025/segmentation_task/train/masks",
    img_size=(256, 256),
    batch_size=32,
    use_augmentation=True
)
```

---

## Download Dataset

The BRISC2025 dataset can be obtained from:
- Official source: [https://www.kaggle.com/datasets/briscdataset/brisc2025/data]

Place the downloaded dataset in the `data/brisc2025/` directory following the structure shown above.

---

## Data Augmentation

### Classification
- Random horizontal/vertical flips
- Random rotations
- Random brightness/contrast adjustments
- Model-specific preprocessing (e.g., ImageNet normalization)

### Segmentation
- Albumentations pipeline with spatial transforms
- Horizontal/vertical flips
- Rotations, shifts, and scaling
- Elastic transforms
- **Note**: Augmentations are applied identically to both image and mask

---

## Raw vs Processed

- **`raw/`**: Original datasets as downloaded (optional backup)
- **`processed/`**: Any custom preprocessing outputs (optional)
- **`brisc2025/`**: Main working dataset directory

Most users will only need the `brisc2025/` directory. The `raw/` and `processed/` directories are optional for advanced workflows.

---

## Dataset Statistics

### Classification Task
- **Total size**: ~3-4 GB
- **Images per class**: ~1,250 train, ~250 test
- **Resolution**: Various (automatically preprocessed)

### Segmentation Task
- **Total size**: ~2-3 GB
- **Image-mask pairs**: Varies by subset
- **Mask coverage**: Variable tumor sizes

---

## Notes

- All scripts automatically handle data loading from the `brisc2025/` directory
- Training data is split into train/validation sets using stratified sampling
- Test data remains held-out for final evaluation
- Ensure sufficient disk space (~6-7 GB total)
