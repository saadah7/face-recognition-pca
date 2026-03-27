import cv2
import numpy as np
from tensorflow.keras.models import load_model
from face_recognition_pca import load_images_from_folder, apply_pca

def preprocess_test_image(path, size=(100, 100)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, size)
    return img_resized.flatten().reshape(-1, 1), img_resized

def predict_face(image_path, k=50):
    data_matrix, labels, label_dict = load_images_from_folder()
    mean_face, feature_vectors, _ = apply_pca(data_matrix, k)

    test_vector, display_img = preprocess_test_image(image_path)
    test_image_centered = test_vector - mean_face
    projected_test = np.dot(feature_vectors.T, test_image_centered).T

    model = load_model("face_recognition_model.h5")
    prediction = model.predict(projected_test)[0]
    predicted_id = np.argmax(prediction)
    confidence = np.max(prediction)
    predicted_name = label_dict[predicted_id]

    print(f"\n🔍 Predicted: {predicted_name} (Confidence: {confidence:.2%})")

    
    display_img_color = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
    cv2.putText(display_img_color, f"{predicted_name} ({confidence:.2%})",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Predicted Face", display_img_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    predict_face(r"C:\Users\Bilal\Pictures\Screenshots\amitabh.png")  
