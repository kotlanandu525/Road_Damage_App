import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import kagglehub
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

path = kagglehub.dataset_download(
    "lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes"
)

base_path = path + "/data"
classes = [
    folder for folder in os.listdir(base_path)
    if os.path.isdir(os.path.join(base_path, folder))
]
IMG_SIZE = 128

X = []
y = []

for label, category in enumerate(classes):

    folder = os.path.join(base_path, category)

    for file in os.listdir(folder):

        try:
            img_path = os.path.join(folder, file)

            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0

            X.append(img)
            y.append(label)

        except:
            pass

X = np.array(X)
y = np.array(y)

y_encoded = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)


model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))

model.compile(
    optimizer='adam',
loss='sparse_categorical_crossentropy',    metrics=['accuracy']
)

model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=5,
    validation_data=(X_test, y_test)
)

loss, accuracy = model.evaluate(X_test, y_test)

predictions = model.predict(X_test)

y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(y_test, axis=1)

report = classification_report(
    y_true,
    y_pred,
    labels=range(len(classes)),
    target_names=classes,
    zero_division=0
)
cm = confusion_matrix(y_true, y_pred)

st.set_page_config(page_title="Road Damage Detection", layout="wide")

st.title("AI-Based Road Damage Detection System")
st.subheader("Smart City Infrastructure Monitoring using CNN")

st.markdown("---")

st.header("About the Project")

st.write("Road monitoring is important for reducing accidents, improving transportation safety, and helping smart cities prioritize infrastructure maintenance.")

st.write("CNN models automatically detect potholes, cracks, and road damage patterns from images using computer vision.")

st.write("Applications include smart transportation systems, municipal road inspection, automated surveillance, and predictive maintenance.")

st.markdown("---")

st.header("Upload Road Image")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.header("Uploaded Image Preview")

    st.image(image, width=500)

    img = np.array(image)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    class_index = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    predicted_class = classes[class_index]

    severity = "Low"

    if confidence > 85:
        severity = "High"
    elif confidence > 60:
        severity = "Medium"

    st.header("Prediction Area")

    st.success(f"Prediction: {predicted_class} Detected")

    st.info(f"Confidence: {confidence:.2f}%")

    st.warning(f"Severity: {severity}")

    st.header("Visualization Area")

    fig, ax = plt.subplots(figsize=(7,4))

    ax.bar(classes, prediction[0])

    ax.set_xlabel("Damage Classes")
    ax.set_ylabel("Confidence Score")
    ax.set_title("Class Confidence Graph")

    st.pyplot(fig)

    st.header("Recommendations")

    if severity == "High":

        st.error("Immediate maintenance recommended.")
        st.error("High-risk road condition detected.")

    elif severity == "Medium":

        st.warning("Road inspection recommended soon.")

    else:

        st.success("Road condition currently manageable.")

st.markdown("---")

st.header("Model Evaluation")

st.write(f"Test Accuracy: {accuracy:.4f}")

st.text(report)

st.subheader("Confusion Matrix")

st.write(cm)
