import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json
import os

# --- Configuration --- #
IMG_SIZE = 128 # Must match the size used during training
MODEL_PATH = 'road_damage_cnn_model.h5'
LABEL_MAP_PATH = 'label_mapping.json'

# --- Load Model and Label Mapping --- #
@st.cache_resource
def load_model_and_labels():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(LABEL_MAP_PATH, 'r') as f:
            label_map_str_keys = json.load(f)
            # Convert string keys back to integers if necessary
            label_map = {int(k) if k.isdigit() else k: v for k, v in label_map_str_keys.items()}
        # Reconstruct class_names and encoder for display purposes
        class_names = [name for _, name in sorted([(v, k) for k, v in label_map.items()])]
        return model, class_names
    except FileNotFoundError:
        st.error(f"Error: Model files not found. Make sure '{MODEL_PATH}' and '{LABEL_MAP_PATH}' are in the same directory as this script.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model or labels: {e}")
        return None, None

model, class_names = load_model_and_labels()

# --- Streamlit App --- #
st.set_page_config(page_title="Road Damage Detector", layout="centered")
st.title("🛣️ AI-Powered Road Damage Detection")
st.write("Upload an image of a road surface to detect potential damage (potholes, cracks, manholes).")

if model is None or class_names is None:
    st.stop() # Stop the app if model loading failed

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1) # Read image in BGR format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) # Resize
    img_normalized = img_resized / 255.0 # Normalize
    img_input = np.expand_dims(img_normalized, axis=0) # Add batch dimension

    # Make prediction
    prediction_probs = model.predict(img_input)
    predicted_class_idx = np.argmax(prediction_probs)
    confidence = np.max(prediction_probs) * 100

    predicted_label = class_names[predicted_class_idx]

    st.success(f"Prediction: **{predicted_label}**")
    st.info(f"Confidence: **{confidence:.2f}%**")

    st.write("--- Recommended Actions ---")
    if "crack" in predicted_label.lower() or "pothole" in predicted_label.lower():
        st.warning("This area likely requires immediate inspection and repair.")
    elif "manhole" in predicted_label.lower():
        st.info("Monitor manhole cover condition and surroundings for further damage.")
    else:
        st.success("Road condition appears good. Continue regular monitoring.")

else:
    st.write("Awaiting image upload for analysis.")

st.sidebar.header("About the App")
st.sidebar.write("This application uses a Convolutional Neural Network (CNN) to classify road conditions from uploaded images.")
st.sidebar.write(f"Model: {MODEL_PATH}")
st.sidebar.write(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
st.sidebar.write(f"Detected Classes: {', '.join(class_names)}")