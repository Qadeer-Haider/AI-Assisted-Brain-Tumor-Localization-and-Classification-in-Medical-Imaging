import os
import tensorflow as tf
import tf2onnx
import glob

def convert_models(base_dir, sub_dir):
    weights_dir = os.path.join(base_dir, 'weights', sub_dir)
    out_dir = os.path.join(base_dir, 'weights', 'onnx', sub_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    keras_files = glob.glob(os.path.join(weights_dir, '*.keras'))
    if not keras_files:
        print(f"No .keras models found in {weights_dir}")
        return

    for keras_file in keras_files:
        model_name = os.path.basename(keras_file).replace('.keras', '.onnx')
        out_path = os.path.join(out_dir, model_name)
        
        if os.path.exists(out_path):
            print(f"Skipping {model_name}, already exists at {out_path}")
            continue

        print(f"Loading {keras_file}...")
        try:
            # We might need safe loading for custom objects
            model = tf.keras.models.load_model(keras_file, compile=False)
            
            print(f"Converting to {out_path}...")
            # Convert to ONNX
            spec = (tf.TensorSpec((None, *model.input_shape[1:]), tf.float32, name="input"),)
            output_path, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=out_path)
            print(f"Successfully saved {output_path}")
        except Exception as e:
            print(f"Failed to convert {keras_file}: {e}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    print("Converting classification models...")
    convert_models(project_root, 'classification')
    
    print("Converting segmentation models...")
    convert_models(project_root, 'segmentation')
    
    print("Done!")
