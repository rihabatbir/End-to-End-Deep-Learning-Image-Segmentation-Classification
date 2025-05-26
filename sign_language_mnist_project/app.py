import streamlit as st
import numpy as np
import os
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
from PIL import Image

# Configuration du modèle
MODEL_REPO = "Roroat/sign-language-model"
MODEL_FILENAME = "sign_language_letters_model.keras"

# 📦 Téléchargement depuis Hugging Face
st.info("⬇️ Téléchargement du modèle depuis Hugging Face...")
try:
model_path = hf_hub_download(
    repo_id="Roroat/sign-language-model",
    filename="sign_language_letters_model.keras",
    local_dir="saved_models",
    revision="main",  # assure que tu es sur la bonne branche
    force_download=True
)
    st.success("✅ Modèle téléchargé avec succès.")
except Exception as e:
    st.error(f"❌ Erreur lors du téléchargement : {e}")
    st.stop()

# 🧠 Chargement du modèle
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"❌ Erreur lors du chargement du modèle : {e}")
    st.stop()

# Classes prédictibles
classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Interface utilisateur
st.title("📷 Reconnaissance de lettres ASL (images réelles)")
uploaded_file = st.file_uploader("Téléversez une image (64x64)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Traitement de l’image
    image = Image.open(uploaded_file).convert('L')
    image = image.resize((64, 64))
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = img_array.reshape(1, 64, 64, 1)

    st.image(image, caption="Image prétraitée", width=150)

    if st.button("🔍 Prédire"):
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_class = classes[predicted_index]
        confidence = np.max(prediction) * 100

        st.success(f"Lettre prédite : **{predicted_class}**")
        st.info(f"🔢 Confiance : {confidence:.2f}%")

        st.subheader("📊 Scores par classe :")
        for i, score in enumerate(prediction[0]):
            st.write(f"{classes[i]} : {score:.4f}")
