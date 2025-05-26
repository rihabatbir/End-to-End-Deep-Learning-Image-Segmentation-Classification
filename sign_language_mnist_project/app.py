import streamlit as st
import numpy as np
import os
import requests
from tensorflow.keras.models import load_model
from PIL import Image

MODEL_PATH = 'saved_models/sign_language_letters_model.keras'
MODEL_URL = 'https://huggingface.co/Roroat/sign-language-model/resolve/main/sign_language_letters_model.keras'

# Télécharger si le modèle n'existe pas
if not os.path.exists(MODEL_PATH):
    os.makedirs("saved_models", exist_ok=True)
    st.info("⬇️ Téléchargement du modèle depuis Hugging Face...")
    r = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    st.success("✅ Modèle téléchargé")

# Charger le modèle après le téléchargement
model = load_model(MODEL_PATH)

# Chargement du modèle entraîné
model = load_model('saved_models/sign_language_letters_model.keras')

# Extraire automatiquement les classes depuis le dossier
classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

st.title("📷 Reconnaissance de lettres ASL (images réelles)")
uploaded_file = st.file_uploader("Téléversez une image (64x64)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Prétraitement de l'image
    image = Image.open(uploaded_file).convert('L')  # niveau de gris
    image = image.resize((64, 64))  # redimensionnement
    img_array = np.array(image).astype(np.float32) / 255.0  # normalisation
    img_array = img_array.reshape(1, 64, 64, 1)  # ajout de la dimension batch

    # Affichage de l'image
    st.image(image, caption='Image prétraitée', width=150)

    if st.button("🔍 Prédire"):
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        predicted_class = classes[predicted_class_index]
        confidence = np.max(prediction) * 100

        st.success(f"Lettre prédite : **{predicted_class}**")
        st.info(f"🔢 Confiance : {confidence:.2f}%")

        # Optionnel : afficher les scores de toutes les classes
        st.subheader("Scores par classe :")
        for i, score in enumerate(prediction[0]):
            st.write(f"{classes[i]} : {score:.4f}")
