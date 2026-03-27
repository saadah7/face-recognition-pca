import numpy as np

def apply_pca(data_matrix, k):
    mean_face = np.mean(data_matrix, axis=1, keepdims=True)
    delta = data_matrix - mean_face

    surrogate_cov = np.dot(delta.T, delta)
    eigvals, eigvecs_small = np.linalg.eigh(surrogate_cov)

    idx = np.argsort(eigvals)[::-1]
    eigvecs_small = eigvecs_small[:, idx]
    eigvals = eigvals[idx]

    eigvecs_large = np.dot(delta, eigvecs_small)
    eigvecs_large = eigvecs_large / np.linalg.norm(eigvecs_large, axis=0)

    feature_vectors = eigvecs_large[:, :k]
    projected_data = np.dot(feature_vectors.T, delta)

    return mean_face, feature_vectors, projected_data
