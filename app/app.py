import streamlit as st
import os
import glob
import onnxruntime as ort
import numpy as np
from PIL import Image
from utils import preprocess_image, decode_classification, postprocess_segmentation, overlay_mask

st.set_page_config(page_title="Brain Tumor Analysis", layout="wide")

# Paths to models inside container/local environment
CLASSIFICATION_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'weights', 'onnx', 'classification')
SEGMENTATION_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'weights', 'onnx', 'segmentation')

def get_available_models(model_dir):
    if not os.path.exists(model_dir):
        return []
    models = glob.glob(os.path.join(model_dir, '*.onnx'))
    return [os.path.basename(m) for m in models]

st.title("🧠 AI-Assisted Brain Tumor Localization & Classification")
st.markdown("Upload an MRI scan to classify the tumor type or segment the tumor region.")

# Sidebar for options
st.sidebar.header("Settings")
task = st.sidebar.radio("Select Task:", ("Classification", "Segmentation"))

model_dir = CLASSIFICATION_DIR if task == "Classification" else SEGMENTATION_DIR
available_models = get_available_models(model_dir)

if not available_models:
    st.sidebar.warning(f"No ONNX models found for {task}. Please ensure the models are converted and placed in the correct directory.")
else:
    selected_model_name = st.sidebar.selectbox("Select Model:", available_models)
    selected_model_path = os.path.join(model_dir, selected_model_name)

st.sidebar.markdown("---")
st.sidebar.info("Choose a model and upload an image to see results.")

# Main upload area
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and available_models:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("")
    st.write("**Analyzing...**")

    try:
        # Load ONNX session
        session = ort.InferenceSession(selected_model_path)
        
        # Get input info
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        # Typically input_shape is [batch, height, width, channels] for Keras translated to ONNX
        h, w = input_shape[1], input_shape[2] 
        target_size = (h, w) if isinstance(h, int) and isinstance(w, int) else (224, 224) 

        # Preprocess
        input_tensor = preprocess_image(image, target_size=target_size, task=task.lower())

        # Run inference
        outputs = session.run(None, {input_name: input_tensor})
        prediction = outputs[0]

        st.subheader("Results:")
        if task == "Classification":
            class_name, confidence = decode_classification(prediction)
            st.success(f"**Predicted Class:** {class_name.capitalize()}")
            st.info(f"**Confidence:** {confidence*100:.2f}%")
            
            # Simple bar chart for all classes
            if prediction.shape[1] == 4:
                from utils import CLASS_NAMES
                import pandas as pd
                df = pd.DataFrame({"Class": CLASS_NAMES, "Confidence": prediction[0]})
                st.bar_chart(df.set_index("Class"))

        elif task == "Segmentation":
            # Segmentation Output processing
            mask = postprocess_segmentation(prediction, image.size)
            
            # Show Mask side-by-side
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(mask, caption="Predicted Mask", use_container_width=True)
                
            with col2:
                blended = overlay_mask(image, mask)
                st.image(blended, caption="Blended Image", use_container_width=True)

    except Exception as e:
        st.error(f"Error during inference: {str(e)}")
elif uploaded_file is None:
    st.info("Please upload an image to begin.")
