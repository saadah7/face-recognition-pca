import os
import cv2
import numpy as np

def load_images_from_folder(folder_path, img_size=(100, 100)):
    images = []
    labels = []
    label_dict = {}
    label_id = 0

    for person_name in sorted(os.listdir(folder_path)):
        person_path = os.path.join(folder_path, person_name)
        if not os.path.isdir(person_path):
            continue

        label_dict[label_id] = person_name

        for filename in os.listdir(person_path):
            img_path = os.path.join(person_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, img_size)
                images.append(img.flatten())
                labels.append(label_id)
        label_id += 1

    return np.array(images).T, np.array(labels), label_dict  # Shape: (mn, p), (p,), dict
