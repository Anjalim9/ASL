import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("asl_model.h5")

# Class labels (must match training order)
class_labels = [
    "A","B","C","D","E","F","G","H","I","J","K","L","M",
    "N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
    "SPACE","DELETE","NOTHING"
]

# Streamlit UI
st.set_page_config(page_title="ASL Detection", layout="centered")
st.title("American Sign Language Detection")

# Choose input method
option = st.radio("Choose input method:", ["Upload from computer", "Capture from camera"])

image = None

if option == "Upload from computer":
    uploaded_file = st.file_uploader("Upload an ASL image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

elif option == "Capture from camera":
    camera_file = st.camera_input("Take a picture")
    if camera_file:
        image = Image.open(camera_file).convert("RGB")

# Run prediction if image is available
if image is not None:
    st.image(image, caption="Input Image", use_container_width=True)

    # Preprocess
    img_size = 64
    img = image.resize((img_size, img_size))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.subheader(f"Prediction: **{predicted_class}**")
    st.write(f"Confidence: {confidence:.2f}")
