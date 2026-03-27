Face Recognition using PCA and ANN
 Overview

This project builds a face recognition system by combining Principal Component Analysis (PCA) for feature extraction and Artificial Neural Networks (ANN) for classification. It reduces image complexity while maintaining important facial features, enabling accurate and efficient recognition.

📁 Project Structure
face_recognition_pca/
│── dataset/                         # Training images (organized by person)
│── face_recognition_pca/            # Python package
│   ├── __init__.py
│   ├── ann_model.py                 # ANN training code
│   ├── pca_module.py                # PCA code
│   ├── utils.py                     # image loading utilities
│── main.py                          # Entrypoint script
│── accuracy_vs_k.py                 # PCA accuracy analysis
│── test_face.py                     # Face recognition test utility
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
1. **Prepare the Dataset:**
   - Ensure your face images are organized in a directory structure like `dataset/person_name/image.jpg`.

2. **Train the Model:**
   - Navigate to the project's root directory in your terminal.
   - Run the training script:
     ```bash
     python accuracy_vs_k.py
     ```
3. **Run Face Recognition on a Test Image:**
   - Modify the `test_face.py` script to point to the image you want to test.
   - Ensure that you have the correct path to the image you want to test.
   - Run the prediction script: `python test_face.py`

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
