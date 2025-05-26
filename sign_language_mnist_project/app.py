import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
from huggingface_hub import hf_hub_download

# --- CONFIGURATION ---
MODEL_PATH = 'saved_models/sign_language_letters_model.keras'

# --- TÉLÉCHARGEMENT DU MODÈLE ---
if not os.path.exists(MODEL_PATH):
    os.makedirs("saved_models", exist_ok=True)
    st.info("⬇️ Téléchargement du modèle depuis Hugging Face...")
    try:
        hf_hub_download(
            repo_id="Roroat/sign-language-model",
            filename="sign_language_letters_model.keras",
            local_dir="saved_models",
            local_dir_use_symlinks=False
        )
        st.success("✅ Modèle téléchargé avec succès.")
    except Exception as e:
        st.error(f"❌ Échec du téléchargement du modèle : {e}")
        st.stop()

# --- CHARGEMENT DU MODÈLE ---
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Erreur lors du chargement du modèle : {e}")
    st.stop()

# --- CLASSES ---
classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# --- INTERFACE UTILISATEUR ---
st.title("📷 Reconnaissance de lettres ASL (images réelles)")
uploaded_file = st.file_uploader("Téléversez une image (64x64)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    image = image.resize((64, 64))
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = img_array.reshape(1, 64, 64, 1)

    st.image(image, caption='Image prétraitée', width=150)

    if st.button("🔍 Prédire"):
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        predicted_class = classes[predicted_class_index]
        confidence = np.max(prediction) * 100

        st.success(f"Lettre prédite : **{predicted_class}**")
        st.info(f"🔢 Confiance : {confidence:.2f}%")

        st.subheader("📊 Scores par classe :")
        for i, score in enumerate(prediction[0]):
            st.write(f"{classes[i]} : {score:.4f}")
