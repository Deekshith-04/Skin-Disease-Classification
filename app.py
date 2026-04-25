import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import gdown
import os

# ================= MODEL DOWNLOAD =================
MODEL_PATH = "skin_model.h5"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=13mkZc07s-6WNldytzgxmvm7mZPweXLk6"
    gdown.download(url, MODEL_PATH, quiet=False)

model = tf.keras.models.load_model(MODEL_PATH)

# ================= CLASS LABELS =================
class_names = ['akiec','bcc','bkl','df','mel','nv','vasc']

# ================= DISEASE INFO =================
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

# ================= UI =================
st.title("Skin Disease Classification System")
st.write("Upload a skin image or scan using camera.")

# ================= HISTORY =================
if "history" not in st.session_state:
    st.session_state.history = []

# ================= INPUT =================
uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])
camera_image = st.camera_input("Scan using Camera")

image_source = None

if uploaded_file is not None:
    image_source = uploaded_file
elif camera_image is not None:
    image_source = camera_image

# ================= PROCESS =================
if image_source is not None:

    image = Image.open(image_source).convert("RGB")
    st.image(image, caption="Selected Image", use_column_width=True)

    image = image.resize((64,64))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Brightness check (FIX for black images)
    brightness = np.mean(img_array)

    if brightness < 0.15:
        st.error("Image too dark ❌ Please capture a clearer skin image.")

    else:
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # Confidence check
        if confidence < 60:
            st.warning("Low confidence ⚠️ Please upload a clearer skin image.")

        else:
            disease_code = class_names[predicted_class]
            info = disease_info[disease_code]

            st.success("Prediction Completed ✅")

            st.write("Disease Code:", disease_code)
            st.write("Full Name:", info["name"])
            st.write("Confidence: {:.2f}%".format(confidence))

            st.subheader("Prescription")
            st.write(info["prescription"])

            st.subheader("Precautions")
            st.write(info["precautions"])

            st.subheader("Prediction Probability")
            st.bar_chart(prediction[0])

            # Save history
            st.session_state.history.append({
                "Disease": info["name"],
                "Confidence": round(confidence,2)
            })

# ================= HISTORY DISPLAY =================
st.markdown("---")
st.subheader("Prediction History")

if st.session_state.history:
    st.table(st.session_state.history)
else:
    st.write("No predictions yet.")

# ================= DATASET INFO =================
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

# ================= MODEL INFO =================
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

# ================= DISCLAIMER =================
st.warning("This system is for educational purposes only and does not replace professional medical diagnosis.")
