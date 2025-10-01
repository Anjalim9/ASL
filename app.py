import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Streamlit page setup
st.set_page_config(page_title="ASL Detection", layout="centered")
st.title("American Sign Language Detection")

# Class labels (must match training order)
class_labels = [
    "A","B","C","D","E","F","G","H","I","J","K","L","M",
    "N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
    "SPACE","DELETE","NOTHING"
]

# -------------------------------
# Load TFLite model (cached)
# -------------------------------
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="asl_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

# Prediction function
def predict_image(image):
    img_size = 64
    img = image.resize((img_size, img_size))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # Set input tensor
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    # Get output and compute predicted class
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = class_labels[np.argmax(output_data)]
    confidence = np.max(output_data)
    return predicted_class, confidence

# -------------------------------
# Streamlit UI
# -------------------------------
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

# Run prediction
if image is not None:
    st.image(image, caption="Input Image", use_container_width=True)
    predicted_class, confidence = predict_image(image)
    st.subheader(f"Prediction: **{predicted_class}**")
    st.write(f"Confidence: {confidence:.2f}")
