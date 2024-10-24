import os
import numpy as np
import pickle
from skimage.feature import hog
import cv2

DATA_DIR = './dataset/train'

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img = cv2.resize(img, (320,180))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_hog, hog_data_img = hog(
            img_gray, pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            orientations=9,
            visualize=True,
            block_norm='L2-Hys')
        data.append(img_hog)
        labels.append(dir_)

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()