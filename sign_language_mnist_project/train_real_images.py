from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import Input
import os

# Préparation des données
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    'data',
    target_size=(64, 64),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    'data',
    target_size=(64, 64),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Détection automatique du nombre de classes
num_classes = train_gen.num_classes
print(f"🔢 Nombre de classes détectées : {num_classes}")

# Définition du modèle
model = Sequential([
    Input(shape=(64, 64, 1)),  # ✅ bonne pratique
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # ✅ flexible
])

# Compilation et entraînement
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=15)

# Sauvegarde du modèle
os.makedirs('saved_models', exist_ok=True)
model.save('saved_models/real_asl_digits_model.keras')
print("✅ Modèle entraîné et sauvegardé avec succès.")
