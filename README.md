Face Recognition using PCA and ANN
 Overview

This project builds a face recognition system by combining Principal Component Analysis (PCA) for feature extraction and Artificial Neural Networks (ANN) for classification. It reduces image complexity while maintaining important facial features, enabling accurate and efficient recognition.

📁 Project Structure
face-recognition-pca-ann/
│── dataset/                # Training images (organized by person)
│── models/                 # Saved trained models
│── src/
│   ├── preprocess.py       # Image preprocessing
│   ├── pca.py              # PCA implementation
│   ├── train_ann.py        # ANN training script
│   ├── predict.py          # Face recognition / testing
│── utils/
│   ├── helpers.py          # Utility functions
│── requirements.txt
│── README.md

⚙️ Technologies Used
Python
OpenCV
NumPy
Scikit-learn / TensorFlow / Keras
Matplotlib

🧠 How It Works
Data Collection
Face images are stored in labeled folders.
Preprocessing
Convert to grayscale
Resize images
Normalize pixel values
Feature Extraction (PCA)
Convert images into Eigenfaces
Reduce dimensionality
Model Training (ANN)
Train neural network on extracted features
Prediction
Input new image
Extract features using PCA
ANN predicts identity

🛠️ Installation
git clone https://github.com/your-username/face-recognition-pca-ann.git
cd face-recognition-pca-ann
pip install -r requirements.txt

▶️ Usage
1. Train the Model
python src/train_ann.py
2. Run Face Recognition
python src/predict.py

📊 Features
Efficient dimensionality reduction using PCA
Accurate classification with ANN
Handles lighting and expression variations
Lightweight and fast

🎯 Applications
Attendance systems
Security & surveillance
Access control
Device authentication

✅ Advantages
Reduced computation time
Better accuracy than traditional methods
Works in real-world conditions
Scalable for larger datasets

🔮 Future Improvements
Replace ANN with CNN for higher accuracy
Real-time webcam integration
GUI-based interface
Cloud deployment

🧾 Conclusion

This project demonstrates how combining PCA and ANN creates a balanced system that is both efficient and accurate for face recognition tasks, making it suitable for practical, real-world use.
