# 📓 Notebooks

This directory contains Jupyter notebooks for training, evaluation, and experimentation.

## Available Notebooks

### 1. `evaluate_all_models.ipynb`
**Purpose**: Comprehensive evaluation of all trained classification models

**Features**:
- 📊 Training curves visualization (loss, accuracy, recall, AUC)
- 🔍 Test set evaluation with detailed metrics
- 📈 Confusion matrices for each model
- 🎨 Beautiful Rich console output
- ⏱️ Progress bars with tqdm
- 📝 Automatic README update with results

**Usage**:
```bash
jupyter notebook notebooks/evaluate_all_models.ipynb
```

**Outputs**:
- Training curve plots in `logs/classification/`
- Confusion matrices for each model
- Test results comparison chart
- Updated README with results table

---

### 2. `train_classification.ipynb`
**Purpose**: Interactive training notebook for classification models

**Features**:
- Quick model training and experimentation
- Hyperparameter tuning
- Real-time visualization of training progress

**Usage**:
```bash
jupyter notebook notebooks/train_classification.ipynb
```

---

### 3. `test_classification.ipynb`
**Purpose**: Testing and debugging classification pipeline

**Features**:
- Data loading tests
- Model architecture verification
- Quick inference tests

---

## Requirements

All notebooks require the following packages:
- TensorFlow 2.10+
- pandas
- matplotlib
- seaborn
- rich (for beautiful console output)
- tqdm (for progress bars)

Install with:
```bash
pip install -r requirements.txt
```

## Tips

- **Restart kernel** if you make changes to `src/` modules
- Use **Rich tables** for displaying results elegantly
- Use **tqdm progress bars** for long-running operations
- Save outputs to avoid re-running expensive evaluations

## Generated Artifacts

Notebooks will generate files in:
- `logs/classification/` - Training logs, plots, confusion matrices
- `weights/classification/` - Model checkpoints
