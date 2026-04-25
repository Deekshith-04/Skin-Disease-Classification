import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("skin_model.h5")

# Class labels
class_names = ['akiec','bcc','bkl','df','mel','nv','vasc']

# Disease information
disease_info = {
    "akiec": {
        "name":"Actinic Keratoses",
        "prescription":"Consult dermatologist. Cryotherapy or topical treatment may be required.",
        "precautions":"Avoid excessive sun exposure. Use sunscreen SPF 50+."
    },
    "bcc": {
        "name":"Basal Cell Carcinoma",
        "prescription":"Medical evaluation required. Surgical removal is common treatment.",
        "precautions":"Avoid UV radiation. Regular skin checkups recommended."
    },
    "bkl": {
        "name":"Benign Keratosis",
        "prescription":"Usually harmless. Monitor for changes.",
        "precautions":"Avoid scratching. Maintain proper skin hygiene."
    },
    "df": {
        "name":"Dermatofibroma",
        "prescription":"Generally no treatment required unless painful.",
        "precautions":"Avoid irritation of the affected area."
    },
    "mel": {
        "name":"Melanoma",
        "prescription":"URGENT dermatologist consultation required.",
        "precautions":"Avoid sun exposure. Seek immediate medical attention."
    },
    "nv": {
        "name":"Melanocytic Nevus",
        "prescription":"Usually a benign mole. Monitor size and color changes.",
        "precautions":"Regular skin examination. Avoid excessive sunlight."
    },
    "vasc": {
        "name":"Vascular Lesion",
        "prescription":"Consult doctor if lesion grows or changes.",
        "precautions":"Protect area from injury."
    }
}

st.title("Skin Disease Classification System")
st.write("Upload a skin image or scan using camera.")

# Prediction history storage
if "history" not in st.session_state:
    st.session_state.history = []

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

# Camera capture
camera_image = st.camera_input("Scan using Camera")

image_source = None

if uploaded_file is not None:
    image_source = uploaded_file
elif camera_image is not None:
    image_source = camera_image

if image_source is not None:

    image = Image.open(image_source).convert("RGB")
    st.image(image, caption="Selected Image", use_column_width=True)

    image = image.resize((64,64))
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    brightness = np.mean(img_array)

    if brightness < 0.15:
        st.error("Image too dark. Please capture a clearer skin image.")

    else:

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        if confidence < 60:
            st.warning("Prediction confidence is low. Please upload a clearer skin image.")

        else:

            disease_code = class_names[predicted_class]
            info = disease_info[disease_code]

            st.success("Prediction Completed")

            st.write("Disease Code:", disease_code)
            st.write("Full Name:", info["name"])
            st.write("Confidence: {:.2f}%".format(confidence))

            st.subheader("Prescription")
            st.write(info["prescription"])

            st.subheader("Precautions")
            st.write(info["precautions"])

            st.subheader("Prediction Probability")
            st.bar_chart(prediction[0])

            # Save prediction history
            st.session_state.history.append({
                "Disease": info["name"],
                "Confidence": round(confidence,2)
            })

# Prediction history display
st.markdown("---")
st.subheader("Prediction History")

if st.session_state.history:
    st.table(st.session_state.history)
else:
    st.write("No predictions yet.")

# Dataset statistics
st.markdown("---")
st.subheader("Dataset Information")

st.write("Dataset: HAM10000")
st.write("Total Images: 10,015")
st.write("Classes: 7 Skin Disease Categories")

dataset_stats = {
    "akiec":327,
    "bcc":514,
    "bkl":1099,
    "df":115,
    "mel":1113,
    "nv":6705,
    "vasc":142
}

st.bar_chart(dataset_stats)

# Model performance
st.markdown("---")
st.subheader("Model Performance")

st.write("Model: Convolutional Neural Network (CNN)")
st.write("Training Epochs: 10")
st.write("Batch Size: 32")
st.write("Image Size: 64x64")

st.write("Accuracy: ~72%")
st.write("Precision: ~0.70")
st.write("Recall: ~0.69")
st.write("F1 Score: ~0.69")

# Disclaimer
st.warning("This system is for educational purposes only and does not replace professional medical diagnosis.")