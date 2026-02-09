#!/usr/bin/env python
"""
Predict tumor segmentation mask for a single image or directory.

Usage:
    python scripts/predict_mask.py --image path/to/image.jpg --model weights/segmentation/UNet_best.keras
    python scripts/predict_mask.py --input-dir path/to/images --model weights/segmentation/UNet_best.keras
"""

import argparse
import glob
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict tumor segmentation masks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--image", "-i",
        type=str,
        default=None,
        help="Path to a single image",
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Path to directory of images",
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to trained model file (.keras)",
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="outputs/predictions",
        help="Directory to save predictions",
    )
    
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Show visualization for each prediction",
    )
    
    parser.add_argument(
        "--save-overlay",
        action="store_true",
        help="Save overlay visualizations",
    )
    
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.5,
        help="Threshold for binary mask (default: 0.5)",
    )
    
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        default=[256, 256],
        help="Image size (height width)",
    )
    
    return parser.parse_args()


def main():
    """Main prediction function."""
    args = parse_args()
    
    if args.image is None and args.input_dir is None:
        print("❌ Error: Must provide either --image or --input-dir")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🧠 Brain Tumor Segmentation Prediction")
    print("=" * 60)
    
    # Import predictor
    from src.segmentation import TumorSegmentor
    
    # Initialize segmentor
    print(f"\n📂 Loading model from: {args.model}")
    segmentor = TumorSegmentor(
        model_path=args.model,
        img_size=tuple(args.img_size),
        threshold=args.threshold,
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of images to process
    if args.image:
        image_paths = [args.image]
    else:
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))
            image_paths.extend(glob.glob(os.path.join(args.input_dir, ext.upper())))
        image_paths = sorted(set(image_paths))
    
    print(f"\n🔍 Processing {len(image_paths)} images...")
    
    # Process each image
    for i, img_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] {os.path.basename(img_path)}")
        
        # Predict
        mask = segmentor.predict(img_path)
        
        # Save mask
        basename = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = output_dir / f"{basename}_mask.png"
        
        import numpy as np
        from PIL import Image
        mask_img = (mask * 255).astype(np.uint8)
        Image.fromarray(mask_img[:, :, 0]).save(mask_path)
        print(f"   💾 Mask saved: {mask_path}")
        
        # Save overlay if requested
        if args.save_overlay:
            overlay_path = output_dir / f"{basename}_overlay.png"
            segmentor.visualize(
                img_path,
                save_path=str(overlay_path),
                show=False,
            )
        
        # Show visualization if requested
        if args.visualize:
            segmentor.visualize(img_path, show=True)
    
    print("\n" + "=" * 60)
    print(f"✅ Complete! {len(image_paths)} images processed.")
    print(f"   Output: {output_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
