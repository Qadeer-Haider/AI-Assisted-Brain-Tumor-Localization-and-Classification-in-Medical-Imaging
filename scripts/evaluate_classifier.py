#!/usr/bin/env python
"""
Evaluate a trained classification model on the test set.

This script loads a trained model and evaluates it on the test dataset,
providing detailed metrics including confusion matrix and per-class performance.

Usage:
    python scripts/evaluate_classifier.py --weights path/to/model.keras
    python scripts/evaluate_classifier.py --weights weights/classification/ConvNeXtBase_best_weights.keras

Examples:
    # Evaluate a trained model
    python scripts/evaluate_classifier.py --weights weights/classification/ConvNeXtBase_best_weights.keras

    # Evaluate with specific model name (for preprocessing)
    python scripts/evaluate_classifier.py --weights model.keras --model ConvNeXtBase

    # Save confusion matrix
    python scripts/evaluate_classifier.py --weights model.keras --save-cm confusion_matrix.png
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained classification model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--weights", "-w",
        type=str,
        required=True,
        help="Path to the trained model weights (.keras file)",
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="ConvNeXtBase",
        help="Model architecture name (for preprocessing selection)",
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    
    parser.add_argument(
        "--save-cm",
        type=str,
        default=None,
        help="Path to save confusion matrix image",
    )
    
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable confusion matrix plotting",
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    import tensorflow as tf
    from src.classification.data import build_dataframe, make_dataset
    from src.classification.inference import ClassificationEvaluator
    
    print("\n" + "=" * 60)
    print("🧠 Brain Tumor Classification Evaluation")
    print("=" * 60)
    print(f"📂 Model: {args.weights}")
    print(f"🏗️ Architecture: {args.model}")
    print("=" * 60 + "\n")
    
    # Load model
    print("📂 Loading model...")
    model = tf.keras.models.load_model(args.weights)
    print("✅ Model loaded successfully!")
    
    # Prepare test dataset
    print("\n📊 Preparing test dataset...")
    test_df = build_dataframe(split="test")
    test_ds = make_dataset(
        test_df,
        model_name=args.model,
        shuffle=False,
        batch_size=args.batch_size,
    )
    print(f"   ↳ Test samples: {len(test_df)}")
    
    # Create evaluator
    evaluator = ClassificationEvaluator(model, test_ds)
    
    # Run evaluation
    print("\n" + "=" * 60)
    print("📊 Running Evaluation")
    print("=" * 60)
    
    results = evaluator.evaluate(verbose=True)
    
    # Print classification report
    evaluator.print_classification_report()
    
    # Plot confusion matrix
    if not args.no_plot:
        evaluator.plot_confusion_matrix(
            normalize=True,
            save_path=args.save_cm,
        )
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
