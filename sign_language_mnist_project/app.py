import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

# Chargement du modèle entraîné
model = load_model('saved_models/sign_language_letters_model.keras')

# Extraire automatiquement les classes depuis le dossier
data_dir = "data"
classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

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
