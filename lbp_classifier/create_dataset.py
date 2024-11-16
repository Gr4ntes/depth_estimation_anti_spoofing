import os
import numpy as np
import pickle
from skimage.feature import hog
from skimage.feature import local_binary_pattern
import cv2

DATA_DIR = '../original_dataset/train'


data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Load the cascade
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # Draw rectangle around the faces and crop the faces
        for (x, y, w, h) in faces:
            faces = gray[y:y + h, x:x + w]

        if faces is not None and len(faces) > 0:
            img_lbp = local_binary_pattern(faces, 10, 5)
            cv2.imwrite("face.jpg", img_lbp)
            n_bins = int(img_lbp.max() + 1)
            hist, _ = np.histogram(img_lbp, density=True, bins=n_bins, range=(0, n_bins))
            data.append(hist)
            labels.append(dir_)

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()