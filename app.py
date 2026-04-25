import streamlit as st
import numpy as np
from PIL import Image
import random

# ================= FAKE MODEL =================
def fake_model_predict(img_array):
    probs = np.random.dirichlet(np.ones(7), size=1)
    return probs

# ================= CLASS LABELS =================
class_names = ['akiec','bcc','bkl','df','mel','nv','vasc']

# ================= DISEASE INFO =================
disease_info = {
    "akiec": {"name":"Actinic Keratoses","prescription":"Consult dermatologist.","precautions":"Use sunscreen."},
    "bcc": {"name":"Basal Cell Carcinoma","prescription":"Medical evaluation required.","precautions":"Avoid UV."},
    "bkl": {"name":"Benign Keratosis","prescription":"Usually harmless.","precautions":"Avoid scratching."},
    "df": {"name":"Dermatofibroma","prescription":"No treatment needed.","precautions":"Avoid irritation."},
    "mel": {"name":"Melanoma","prescription":"URGENT consultation required.","precautions":"Avoid sun."},
    "nv": {"name":"Nevus","prescription":"Monitor changes.","precautions":"Skin checkups."},
    "vasc": {"name":"Vascular Lesion","prescription":"Consult doctor.","precautions":"Protect area."}
}

# ================= UI =================
st.title("Skin Disease Classification System")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])
camera_image = st.camera_input("Use Camera")

image_source = uploaded_file if uploaded_file else camera_image

if image_source:

    image = Image.open(image_source).convert("RGB")
    st.image(image, caption="Selected Image")

    image = image.resize((64,64))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    brightness = np.mean(img_array)

    if brightness < 0.15:
        st.error("Image too dark ❌")

    else:
        prediction = fake_model_predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        if confidence < 60:
            st.warning("Low confidence ⚠️")

        else:
            disease_code = class_names[predicted_class]
            info = disease_info[disease_code]

            st.success("Prediction Done ✅")
            st.write("Disease:", info["name"])
            st.write("Confidence:", round(confidence,2), "%")

            st.subheader("Prescription")
            st.write(info["prescription"])

            st.subheader("Precautions")
            st.write(info["precautions"])
