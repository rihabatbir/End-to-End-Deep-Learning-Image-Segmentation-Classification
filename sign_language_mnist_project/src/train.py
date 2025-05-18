from data_loader import load_data
from model import create_model
import os

# Charger uniquement les données d'entraînement
train_gen = load_data(data_dir='data')

# Détecter le nombre de classes
num_classes = train_gen.num_classes
print(f"🧠 Classes détectées : {num_classes}")
print("🔢 Images d'entraînement :", train_gen.samples)

# Créer le modèle
model = create_model(num_classes=num_classes)

# Entraîner
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, epochs=15)

# Sauvegarder le modèle
os.makedirs('saved_models', exist_ok=True)
model.save('saved_models/sign_language_letters_model.keras')
print("✅ Modèle entraîné et sauvegardé avec succès.")
