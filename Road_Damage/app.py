import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json

from PIL import Image

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🚧",
    layout="centered"
)

# =========================================================
# TITLE
# =========================================================

st.title("🚧 Road Damage Detection System")

st.write(
    "Upload a road image to detect potholes, cracks, and manholes."
)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "road_damage_cnn_model.h5"
    )

    return model

model = load_model()

# =========================================================
# LOAD LABEL MAPPING
# =========================================================

with open("label_mapping.json", "r") as f:

    label_map = json.load(f)

# Convert dict keys to list
class_names = list(label_map.keys())

# =========================================================
# IMAGE SIZE
# =========================================================

IMG_SIZE = 128

# =========================================================
# PREPROCESS FUNCTION
# =========================================================

def preprocess_image(image):

    image = np.array(image)

    # Convert RGB if needed
    if image.shape[-1] == 4:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGBA2RGB
        )

    # Resize
    image = cv2.resize(
        image,
        (IMG_SIZE, IMG_SIZE)
    )

    # Normalize
    image = image / 255.0

    # Expand dimensions
    image = np.expand_dims(image, axis=0)

    return image

# =========================================================
# FILE UPLOADER
# =========================================================

uploaded_file = st.file_uploader(
    "Upload Road Image",
    type=["jpg", "jpeg", "png"]
)

# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        width="stretch"
    )

    with st.spinner("Analyzing road condition..."):

        processed_image = preprocess_image(image)

        prediction = model.predict(processed_image)

        predicted_class = np.argmax(prediction)

        confidence = np.max(prediction)

        predicted_label = class_names[predicted_class]

    # =====================================================
    # RESULT
    # =====================================================

    st.success(
        f"Prediction: {predicted_label}"
    )

    st.info(
        f"Confidence: {confidence:.2f}"
    )

    # =====================================================
    # PROBABILITY SCORES
    # =====================================================

    st.subheader("Prediction Probabilities")

    for i, prob in enumerate(prediction[0]):

        st.write(
            f"{class_names[i]} : {prob:.2f}"
        )

        st.progress(float(prob))

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "AI-powered Smart Road Damage Detection using CNN"
)
