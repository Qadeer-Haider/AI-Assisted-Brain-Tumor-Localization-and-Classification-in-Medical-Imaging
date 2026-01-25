#!/usr/bin/env python
"""
Train a brain tumor classification model.

This script trains a classification model using the specified configuration.
It handles data loading, model building, training, and saves the best weights.

Usage:
    python scripts/train_classifier.py
    python scripts/train_classifier.py --config configs/classification_config.yaml
    python scripts/train_classifier.py --model ConvNeXtBase --epochs 100

Examples:
    # Train with default ConvNeXtBase model
    python scripts/train_classifier.py

    # Train with specific model
    python scripts/train_classifier.py --model ResNet152V2

    # Train with custom config
    python scripts/train_classifier.py --config configs/my_config.yaml
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
        description="Train a brain tumor classification model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Model architecture (overrides config)",
    )
    
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)",
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    
    parser.add_argument(
        "--learning-rate", "-lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Disable data augmentation",
    )
    
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class weight balancing",
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Import after adding to path
    from src.utils.config import load_config
    from src.classification.training import ClassificationTrainer
    
    # Load configuration
    config = load_config(args.config, task="classification")
    
    # Override config with command line arguments
    if args.model:
        config["model_name"] = args.model
    if args.epochs:
        config["epochs"] = args.epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.learning_rate:
        config["learning_rate"] = args.learning_rate
    if args.no_augmentation:
        config["augmentation"]["enabled"] = False
    if args.no_class_weights:
        config["use_class_weights"] = False
    
    # Print configuration
    print("\n" + "=" * 60)
    print("🧠 Brain Tumor Classification Training")
    print("=" * 60)
    print(f"📦 Model: {config['model_name']}")
    print(f"📊 Batch Size: {config['batch_size']}")
    print(f"📈 Learning Rate: {config['learning_rate']}")
    print(f"🔄 Epochs: {config['epochs']}")
    print(f"✨ Augmentation: {config.get('augmentation', {}).get('enabled', True)}")
    print(f"⚖️ Class Weights: {config.get('use_class_weights', True)}")
    print("=" * 60 + "\n")
    
    # Create trainer
    trainer = ClassificationTrainer(
        model_name=config["model_name"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        config=config,
    )
    
    # Prepare data
    trainer.prepare_data(
        val_split=config.get("val_split", 0.2),
        use_augmentation=config.get("augmentation", {}).get("enabled", True),
    )
    
    # Build and compile model
    trainer.build()
    trainer.compile()
    
    # Train
    trainer.train(
        epochs=config["epochs"],
        use_class_weights=config.get("use_class_weights", True),
    )
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("📊 Evaluating on Test Set")
    print("=" * 60)
    trainer.evaluate()
    
    # Save final model
    trainer.save()
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
