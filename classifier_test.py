import pickle
import numpy as np
from skimage.feature import hog
import cv2
import torch
import matplotlib.pyplot as plt

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        # Capture the image
        img = frame.copy()

        # Process the image with midas
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
        output_filename = "temp_img.png"
        fig = plt.figure()
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(output)
        plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        img = cv2.imread(output_filename)
        img = cv2.resize(img, (320,180))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(img_gray.shape)

        # classify the image
        img_hog, hog_data_img = hog(
            img_gray, pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            orientations=9,
            visualize=True,
            block_norm='L2-Hys')
        plt.imshow(hog_data_img)
        plt.show()
        input_img = [np.asarray(img_hog)]
        prediction = model.predict(input_img)

        predicted_result = prediction[0]
        print(predicted_result)

