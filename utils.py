import os
import cv2
import pickle
import torch
import inspect
from matplotlib import pyplot as plt
import face_recognition
import numpy as np
from skimage.feature import hog

dir_path = os.path.dirname(__file__)
model_dict = pickle.load(open(os.path.join(dir_path, "model.p"), 'rb'))
model = model_dict['model']

model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

def identify_face(dir, img_path):
    files = os.listdir(dir)
    unknown_image = face_recognition.load_image_file(img_path)
    if len(face_recognition.face_encodings(unknown_image)) > 0:
        unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
    else:
        return False
    for file in files:
        known_image = face_recognition.load_image_file(os.path.join(dir, file))
        known_encoding = face_recognition.face_encodings(known_image)[0]
        results = face_recognition.compare_faces([known_encoding], unknown_encoding)
        if results[0] == np.True_:
            return True
    return False

def validate(original_img_path, depth_img_path, hog_img_path, save_images):
    img = cv2.imread(original_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()

    fig = plt.figure()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(output)
    plt.savefig(depth_img_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    img = cv2.imread(depth_img_path)
    img = cv2.resize(img, (320, 180))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # classify the image
    img_hog, hog_data_img = hog(
        img_gray, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        orientations=9,
        visualize=True,
        block_norm='L2-Hys')
    if save_images:
        fig = plt.figure()
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(hog_data_img)
        plt.savefig(hog_img_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    input_img = [np.asarray(img_hog)]
    prediction = model.predict(input_img)
    predicted_result = prediction[0]
    if not save_images:
        os.remove(depth_img_path)
    return predicted_result