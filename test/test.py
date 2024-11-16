import os
import sys
import time

import numpy as np

# fixing path for loading the model from the utils script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import utils

DATA_DIR = './test_images'
depth_img_path = "./depth_img.png"
hog_img_path = "./hog_img.png"

mistakes_count = 0
times = []
for dir_ in os.listdir(DATA_DIR):
    if not os.path.isfile(os.path.join(DATA_DIR, dir_)):
        for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
            start_time = time.perf_counter()
            result = utils.validate(os.path.abspath(os.path.join(DATA_DIR, dir_, img_path)), depth_img_path, hog_img_path, False)
            if result != dir_:
                print(os.path.join(DATA_DIR, dir_, img_path))
                mistakes_count += 1
            end_time = time.perf_counter()
            times.append(end_time - start_time)

print("Mistakes: {}".format(mistakes_count))
print(len(times))
print(np.average(times))
