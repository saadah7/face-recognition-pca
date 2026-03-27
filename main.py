from face_recognition_pca import load_images_from_folder, apply_pca, train_ann

def main():
    data_matrix, labels, label_dict = load_images_from_folder()
    k = 50  # Number of eigenfaces

    mean_face, feature_vectors, projected_data = apply_pca(data_matrix, k)
    model, test_accuracy = train_ann(projected_data, labels, num_classes=len(label_dict))

    print(f"\n✅ Face recognition ANN trained. Test Accuracy = {test_accuracy:.2%}")
    model.save("face_recognition_model.h5")
    print("✅ Model saved as 'face_recognition_model.h5'")

if __name__ == "__main__":
    main()
