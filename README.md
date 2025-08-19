# ✨ Handwritten Alphabet Recognition with EMNIST + Gradio  

This project is an **interactive AI application** that recognizes **hand-drawn English capital letters (A–Z)**.  
It combines **deep learning (CNNs with TensorFlow)** and a **user-friendly Gradio web app** to make handwriting recognition accessible and fun.  

---

## 📚 Overview  

The project has two main parts:  

1. **Model Training**  
   - Trains a Convolutional Neural Network (CNN) on the [EMNIST Letters dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset).  
   - Includes both a **baseline model** and an **improved model** with augmentation, dropout, and batch normalization.  

2. **Interactive App**  
   - A Gradio web app where users can **draw letters in the browser** and get instant predictions.  
   - Preprocessing is applied (grayscale conversion, resizing, normalization) to match EMNIST input format.  

---

## 📂 Project Structure  

├── initial_app.py # Minimal Gradio app using baseline model
├── app.py # Improved Gradio app with preprocessing + better UX
├── improved_model.py # Training script for CNN
├── model.keras # Baseline trained model (simple CNN)
├── improved_model.py # Training script for improved CNN
├── improved_model.keras # Improved model with augmentation + callbacks
└── README.md # Project documentation

## 🚀 Features  

- 🖌️ **Sketch a letter** directly in your browser.  
- 🔍 **Smart preprocessing** (denoising, thresholding, contour cropping).  
- ⚡ **Instant predictions** with top-5 most likely letters + probabilities.  
- 🧠 **Improved CNN architecture** with data augmentation for better accuracy.  
- 🌍 **Deployable anywhere** with Gradio (local or Hugging Face Spaces).  

