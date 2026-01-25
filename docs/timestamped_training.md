# Timestamped Training Runs

## How It Works

When you train a model, the system now **automatically checks** if weights already exist:

### **First Training Run**
```
weights/classification/
├── ConvNeXtBase_best_weights.keras  ← Saved here

logs/classification/
├── ConvNeXtBase_training_log.csv    ← Saved here
```

### **Second Training Run (Same Model)**
When you train `ConvNeXtBase` again, the system detects existing weights and creates a **timestamped folder**:

```
weights/classification/
├── ConvNeXtBase_best_weights.keras          ← Original (preserved)
└── ConvNeXtBase_20260125_203000/            ← New timestamped folder
    └── ConvNeXtBase_best_weights.keras

logs/classification/
├── ConvNeXtBase_training_log.csv            ← Original (preserved)
└── ConvNeXtBase_20260125_203000/            ← New timestamped folder
    └── ConvNeXtBase_training_log.csv
```

### **Multiple Training Runs**
Each subsequent run creates a new timestamped folder:

```
weights/classification/
├── ConvNeXtBase_best_weights.keras          ← Run 1 (original)
├── ConvNeXtBase_20260125_203000/            ← Run 2
│   └── ConvNeXtBase_best_weights.keras
├── ConvNeXtBase_20260125_215430/            ← Run 3
│   └── ConvNeXtBase_best_weights.keras
└── ConvNeXtBase_20260126_094512/            ← Run 4
    └── ConvNeXtBase_best_weights.keras
```

---

## Benefits

✅ **No Overwriting** - All training runs preserved  
✅ **Easy Comparison** - Compare different runs  
✅ **Experiment Tracking** - Keep history of all experiments  
✅ **Automatic** - No manual folder creation needed  

---

## Timestamp Format

`{ModelName}_{YYYYMMDD}_{HHMMSS}`

Example: `ConvNeXtBase_20260125_203045`
- Date: 2026-01-25
- Time: 20:30:45 (8:30:45 PM)

---

## Training Usage

### Command Line
```bash
# First run
python scripts/train_classifier.py --model ConvNeXtBase --epochs 100

# Second run (automatically creates timestamped folder)
python scripts/train_classifier.py --model ConvNeXtBase --epochs 100
```

You'll see:
```
⚠️ Weights already exist! Creating new run folder: ConvNeXtBase_20260125_203000
```

### Python API
```python
from src.classification import ClassificationTrainer

# Train once
trainer1 = ClassificationTrainer(model_name="ConvNeXtBase")
trainer1.prepare_data()
trainer1.build()
trainer1.compile()
trainer1.train(epochs=100)

# Train again with different settings
trainer2 = ClassificationTrainer(model_name="ConvNeXtBase")
trainer2.prepare_data()  # Different split or augmentation
trainer2.build()
trainer2.compile()
trainer2.train(epochs=50)  # New timestamped folder created automatically
```

---

## Finding Your Weights

### Latest Run (Timestamped Folder)
```python
# The trainer stores the checkpoint path
print(trainer.checkpoint_path)
# Output: weights/classification/ConvNeXtBase_20260125_203000/ConvNeXtBase_best_weights.keras
```

### Original Run (Base Folder)
```
weights/classification/ConvNeXtBase_best_weights.keras
```

---

## Loading Models

### Load Latest Timestamped Model
```python
from src.classification.inference import TumorClassifier

classifier = TumorClassifier(
    "weights/classification/ConvNeXtBase_20260125_203000/ConvNeXtBase_best_weights.keras"
)
```

### Load Original Model
```python
classifier = TumorClassifier(
    "weights/classification/ConvNeXtBase_best_weights.keras"
)
```

---

## Tips

🔍 **Compare Runs**: Use CSV logs to compare different training experiments  
📊 **Best Run**: Pick the best performing run based on validation metrics  
🗑️ **Cleanup**: Delete timestamped folders for failed/poor experiments  
📁 **Organize**: Keep original run as your "production" model
