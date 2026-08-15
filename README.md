# 🤟 Sign Language Letter Recognition — End-to-End Deep Learning

An end-to-end deep learning project for recognizing American Sign Language (ASL) hand-signed letters from images, covering the full pipeline from data preparation to a deployed interactive web app.

---

## 📌 Project Overview

This project builds an image classification model that recognizes ASL letters (A–Z) from photos of hand signs, and deploys it as an interactive **Streamlit** web application.

Unlike a simple training notebook, the project follows a complete end-to-end workflow:

- Data preparation and preprocessing
- CNN model design and training
- Model evaluation (classification report, confusion matrix)
- Model hosting on **Hugging Face Hub**
- Deployment as an interactive web app with **Streamlit**

---

## 🎯 Objectives

- Build an image classification pipeline for hand sign recognition.
- Design and train a Convolutional Neural Network (CNN) from scratch.
- Evaluate model performance with standard classification metrics.
- Package and host the trained model externally (Hugging Face Hub) instead of committing large binary files to Git.
- Deploy a usable, interactive demo application.

---

## 🧠 Model

A custom CNN built with `tensorflow.keras`:

```text
Input (64x64, grayscale)
   │
   ▼
Conv2D(32) → MaxPooling2D
   │
   ▼
Conv2D(64) → MaxPooling2D
   │
   ▼
Flatten → Dense(128) → Dropout(0.5)
   │
   ▼
Dense(num_classes, softmax)
```

- Input: 64×64 grayscale images
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Trained with an 80/20 train/validation split (`ImageDataGenerator`)

---

## 📂 Project Structure

```text
sign_language_mnist_project/
│
├── app.py                     # Streamlit web app (inference)
├── train_real_images.py       # Training script (real photo dataset)
├── requirements.txt
├── runtime.txt
│
├── src/
│   ├── model.py                # CNN architecture definition
│   ├── data_loader.py          # Data loading utilities
│   ├── train.py                # Training pipeline
│   └── evaluate.py             # Evaluation: classification report + confusion matrix
│
├── notebook/
│   └── sign_language_training.ipynb   # Exploratory training notebook
│
├── data/                       # Training images, organized by class (A, B, C, ...)
├── saved_models/               # Local model checkpoint (also hosted on Hugging Face Hub)
└── .devcontainer/
```

---

## ⚙️ Installation

```bash
git clone https://github.com/rihabatbir/End-to-End-Deep-Learning-Image-Segmentation-Classification.git
cd End-to-End-Deep-Learning-Image-Segmentation-Classification/sign_language_mnist_project
pip install -r requirements.txt
```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

The app downloads the trained model automatically from the [Hugging Face Hub](https://huggingface.co/Roroat/sign-language-model) on first launch, then lets you upload a hand-sign image and get a predicted letter with a confidence score.

---

## 🏋️ Training Your Own Model

```bash
python train_real_images.py
```

This trains the CNN on the images in `data/` (organized in one subfolder per letter) and saves the resulting model to `saved_models/`.

Evaluation (classification report + confusion matrix) is available via `src/evaluate.py`.

---

## 🛠️ Technologies

- **Deep Learning:** TensorFlow / Keras
- **Web App:** Streamlit
- **Model Hosting:** Hugging Face Hub
- **Data & Evaluation:** NumPy, Pillow, scikit-learn, seaborn, matplotlib

---

## 📊 Results

*(Add your final metrics here — accuracy, precision/recall per class, or the confusion matrix screenshot — to make this section complete.)*

| Metric | Value |
|---|---|
| Validation Accuracy | — |
| Number of Classes | 26 (A–Z) |
| Input Size | 64×64 grayscale |

---

## ⚠️ Limitations

- The model is trained on a limited, self-collected image dataset — generalization to varied lighting, backgrounds, or hand shapes is not guaranteed.
- Real-time video recognition is not implemented; the app works on single uploaded images.
- Dynamic/motion-based signs are not covered — only static letter signs.

---

## 🚀 Possible Improvements

- Expand the training dataset for better generalization.
- Add data augmentation (rotation, brightness, occlusion).
- Support real-time webcam inference.
- Extend to full ASL vocabulary (words, not just letters).

---

## 👩‍💻 Author

**Rihab Atbir**
Master's Student in Computer Science and Telecommunications
Specialization: Applied Artificial Intelligence
Mohammed V University in Rabat — Faculty of Sciences
