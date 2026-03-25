import matplotlib.pyplot as plt
from utils import load_images_from_folder
from pca_module import apply_pca
from ann_model import train_ann

def evaluate_accuracy_vs_k(k_values):
    data_matrix, labels, label_dict = load_images_from_folder(r"C:\Users\ab050\OneDrive\Desktop\internship_project\face_recognition_pca\dataset")
    accuracies = []

    for k in k_values:
        print(f"Testing with k = {k}")
        _, _, projected_data = apply_pca(data_matrix, k)
        _, acc = train_ann(projected_data, labels, num_classes=len(label_dict))
        accuracies.append(acc)

    return k_values, accuracies

if __name__ == "__main__":
    k_vals = [10, 20, 30, 40, 50, 60, 80, 100]
    k_values, accs = evaluate_accuracy_vs_k(k_vals)

    plt.plot(k_values, accs, marker='o')
    plt.xlabel("k (Number of Eigenfaces)")
    plt.ylabel("Test Accuracy")
    plt.title("Accuracy vs Number of Eigenfaces")
    plt.grid(True)
    plt.show()
