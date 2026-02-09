#!/usr/bin/env python
"""
Train a brain tumor segmentation model.

Supports multiple architectures (UNet, AttentionUNet, ResUNetPP, SwinUNet)
and loss functions (dice, bce_tversky, tversky, focal_tversky).

Usage:
    python scripts/train_segmentor.py
    python scripts/train_segmentor.py --model AttentionUNet --loss bce_tversky
    python scripts/train_segmentor.py --config configs/segmentation_config.yaml
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a brain tumor segmentation model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with defaults (UNet + BCE-Tversky loss)
  python scripts/train_segmentor.py

  # Train Attention U-Net
  python scripts/train_segmentor.py --model AttentionUNet

  # Train with specific loss
  python scripts/train_segmentor.py --loss focal_tversky

  # Train with config file
  python scripts/train_segmentor.py --config configs/segmentation_config.yaml
        """,
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
        default="UNet",
        choices=["UNet", "AttentionUNet", "ResUNetPP", "SwinUNet"],
        help="Model architecture",
    )
    
    parser.add_argument(
        "--loss", "-l",
        type=str,
        default="bce_tversky",
        choices=["dice", "dice_bce", "tversky", "focal_tversky", "bce_tversky"],
        help="Loss function",
    )
    
    parser.add_argument(
        "--backbone", "-b",
        type=str,
        default="ResNet50",
        help="Backbone architecture for encoder",
    )
    
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=200,
        help="Number of training epochs",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    
    parser.add_argument(
        "--learning-rate", "--lr",
        type=float,
        default=1e-4,
        help="Initial learning rate",
    )
    
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        default=[256, 256],
        help="Image size (height width)",
    )
    
    parser.add_argument(
        "--train-images",
        type=str,
        default="data/brisc2025/segmentation_task/train/images",
        help="Path to training images directory",
    )
    
    parser.add_argument(
        "--train-masks",
        type=str,
        default="data/brisc2025/segmentation_task/train/masks",
        help="Path to training masks directory",
    )
    
    parser.add_argument(
        "--test-images",
        type=str,
        default="data/brisc2025/segmentation_task/test/images",
        help="Path to test images directory",
    )
    
    parser.add_argument(
        "--test-masks",
        type=str,
        default=None,
        help="Path to test masks directory (if available)",
    )
    
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split fraction",
    )
    
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Disable data augmentation",
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    """Main training function."""
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("🧠 Brain Tumor Segmentation Training")
    print("=" * 60)
    
    # Load config if provided
    config = {}
    if args.config:
        print(f"\n📄 Loading config from: {args.config}")
        config = load_config(args.config)
    
    # Override config with CLI args
    model_name = config.get("model_name", args.model)
    loss_name = config.get("loss", args.loss)
    backbone = config.get("backbone", args.backbone)
    epochs = config.get("epochs", args.epochs)
    batch_size = config.get("batch_size", args.batch_size)
    learning_rate = config.get("learning_rate", args.learning_rate)
    img_size = tuple(config.get("img_size", args.img_size))
    
    # Data paths from config or args
    data_config = config.get("data", {})
    train_images = data_config.get("train_images_dir", args.train_images)
    train_masks = data_config.get("train_masks_dir", args.train_masks)
    test_images = data_config.get("test_images_dir", args.test_images)
    test_masks = data_config.get("test_masks_dir", args.test_masks)
    val_split = data_config.get("val_split", args.val_split)
    
    # Print configuration
    print(f"\n🔧 Configuration:")
    print(f"   Model:         {model_name}")
    print(f"   Backbone:      {backbone}")
    print(f"   Loss:          {loss_name}")
    print(f"   Epochs:        {epochs}")
    print(f"   Batch size:    {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Image size:    {img_size}")
    print(f"   Augmentation:  {'Disabled' if args.no_augmentation else 'Enabled'}")
    
    # Import and train
    from src.segmentation import SegmentationTrainer
    
    trainer = SegmentationTrainer(
        model_name=model_name,
        loss_name=loss_name,
        img_size=img_size,
        batch_size=batch_size,
        learning_rate=learning_rate,
        backbone=backbone,
    )
    
    # Prepare data
    trainer.prepare_data(
        train_images_dir=train_images,
        train_masks_dir=train_masks,
        test_images_dir=test_images,
        test_masks_dir=test_masks,
        val_split=val_split,
        use_augmentation=not args.no_augmentation,
    )
    
    # Build and compile
    trainer.build()
    trainer.compile()
    
    # Train
    history = trainer.train(epochs=epochs)
    
    # Evaluate if test data available
    if test_masks:
        results = trainer.evaluate()
        print("\n📊 Test Results:")
        for metric, value in results.items():
            print(f"   {metric}: {value:.4f}")
    
    # Save final model
    trainer.save()
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
