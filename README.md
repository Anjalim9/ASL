# 🖐️ ASL Detection (American Sign Language)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://streamlit.io/)

---
<img width="510" height="653" alt="{20713B14-8A41-4759-A113-7DC277022A71}" src="https://github.com/user-attachments/assets/77fa3268-319c-4d83-922c-79222f213465" />

This project is an end-to-end system for detecting **American Sign Language (ASL)** hand signs from images.  
It uses a **Convolutional Neural Network (CNN)** to classify hand gestures into **29 classes**:  
- 26 letters (A–Z)  
- 3 special tokens: **SPACE, DELETE, NOTHING**

The trained model is integrated into a **Streamlit web app**, allowing users to make real-time predictions via:
- 📷 Webcam capture  
- 📂 Image upload  

---
## 🚀 Features
- Deep learning–based hand sign classification (29 classes)  
- Data preprocessing (normalization, augmentation, train/test split)  
- Streamlit UI for an interactive experience  
- Supports **camera input** and **file uploads**  
- Easily deployable on **Streamlit Cloud / local machine**  

---
## ⚙️ Installation & Setup
1. Clone this repository:
   
   git clone https://github.com/Anjalim9/ASL.git<br>
   cd ASL
2. Create a virtual environment:

   python -m venv venv<br>
   venv\Scripts\activate   # On Windows<br>
   source venv/bin/activate # On Mac/Linux<br>
3. Install dependencies:

   pip install -r requirements.txt
4. Start the Streamlit app with:

   streamlit run app.py
---   
🧪 Dataset

Dataset Link: [ASL Alphabet Dataset (Kaggle)](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

The dataset contains 29 classes (A–Z, SPACE, DELETE, NOTHING).

Images are organized in class-wise folders.

Preprocessing steps include resizing, normalization, and data augmentation.   
---
📊 Model

Architecture: Convolutional Neural Network (CNN)

Framework: TensorFlow / Keras

Input Size: 64×64 RGB images

Evaluation Metrics: Accuracy, loss
