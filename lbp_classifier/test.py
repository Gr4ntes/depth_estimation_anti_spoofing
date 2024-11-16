import os
import sys
import time
import pickle
import cv2
from skimage.feature import local_binary_pattern

import numpy as np

model_dict = pickle.load(open('./model_lbp.p', 'rb'))
model = model_dict['model']

DATA_DIR = '/Users/gr4ntes/PycharmProjects/depth_estimation_anti_spoofing/test/test_images'

mistakes_count = 0
times = []
for dir_ in os.listdir(DATA_DIR):
    if not os.path.isfile(os.path.join(DATA_DIR, dir_)):
        for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
            start_time = time.perf_counter()
            img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Load the cascade
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            # Draw rectangle around the faces and crop the faces
            for (x, y, w, h) in faces:
                faces = gray[y:y + h, x:x + w]

            result = ""
            if faces is not None and len(faces) > 0:
                img_lbp = local_binary_pattern(faces, 10, 5)
                cv2.imwrite("face.jpg", img_lbp)
                n_bins = int(img_lbp.max() + 1)
                hist, _ = np.histogram(img_lbp, density=True, bins=n_bins, range=(0, n_bins))
                input_img = [np.asarray(hist)]
                result = model.predict(input_img)
            if result != dir_:
                print(os.path.join(DATA_DIR, dir_, img_path))
                mistakes_count += 1
            end_time = time.perf_counter()
            times.append(end_time - start_time)

print("Mistakes: {}".format(mistakes_count))
print(len(times))
print(np.average(times))