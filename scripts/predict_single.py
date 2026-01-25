#!/usr/bin/env python
"""
Predict brain tumor class for a single image.

This script loads a trained model and predicts the tumor class
for a given MRI image, displaying the result with confidence scores.

Usage:
    python scripts/predict_single.py --image path/to/image.jpg --weights path/to/model.keras

Examples:
    # Basic prediction
    python scripts/predict_single.py --image data/brisc2025/classification_task/test/glioma/image.jpg --weights weights/classification/ConvNeXtBase_best_weights.keras

    # Prediction without displaying image
    python scripts/predict_single.py --image image.jpg --weights model.keras --no-display

    # Get top-3 predictions
    python scripts/predict_single.py --image image.jpg --weights model.keras --top-k 3
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
        description="Predict brain tumor class for a single image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="Path to the input image",
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
        "--top-k", "-k",
        type=int,
        default=1,
        help="Show top-k predictions",
    )
    
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display the image",
    )
    
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save the prediction visualization",
    )
    
    return parser.parse_args()


def main():
    """Main prediction function."""
    args = parse_args()
    
    from src.classification.inference import TumorClassifier
    
    print("\n" + "=" * 60)
    print("🧠 Brain Tumor Prediction")
    print("=" * 60)
    print(f"📷 Image: {args.image}")
    print(f"📂 Model: {args.weights}")
    print("=" * 60 + "\n")
    
    # Create classifier
    classifier = TumorClassifier(
        model_path=args.weights,
        model_name=args.model,
    )
    
    # Make prediction
    result = classifier.predict(args.image, return_probabilities=True)
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 Prediction Results")
    print("=" * 60)
    print(f"✅ Predicted Class: {result['class']}")
    print(f"📈 Confidence: {result['confidence']:.2f}%")
    print(f"\n{result['warning']}")
    
    # Show top-k if requested
    if args.top_k > 1:
        print(f"\n🏆 Top-{args.top_k} Predictions:")
        top_k = classifier.get_top_k_predictions(args.image, k=args.top_k)
        for i, (cls, conf) in enumerate(top_k, 1):
            print(f"   {i}. {cls}: {conf:.2f}%")
    
    # Display image if requested
    if not args.no_display:
        import matplotlib.pyplot as plt
        from tensorflow.keras.preprocessing import image as keras_image
        
        img = keras_image.load_img(args.image, target_size=(224, 224))
        
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.axis("off")
        plt.title(
            f"Prediction: {result['class']} ({result['confidence']:.1f}%)",
            fontsize=14,
            fontweight="bold",
        )
        
        if args.save:
            plt.savefig(args.save, dpi=150, bbox_inches="tight")
            print(f"\n💾 Visualization saved to: {args.save}")
        
        plt.show()
    
    print("\n✅ Prediction complete!")


if __name__ == "__main__":
    main()
