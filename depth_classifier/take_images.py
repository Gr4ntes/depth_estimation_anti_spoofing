import os
import cv2
import torch
import numpy as np

import matplotlib.pyplot as plt

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

output_dir = "../dataset/train/fake"
# Specify the valid directory
os.makedirs(output_dir, exist_ok=True)

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

timestamp = 400

while timestamp < 500:
    ret, frame = cap.read()
    if not ret:
        continue
    cv2.imshow("Frame", frame)
    #key = cv2.waitKey(1) & 0xFF
    #if key == ord('s'):
    # Capture the image
    img = frame.copy()

    # Process the image
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

    # Save the valid image
    #output_image = valid * 255
    # Normalize valid to [0, 255]
    #output_image = output_image.astype(np.uint8)
    output_filename = os.path.join(output_dir, f"output_{timestamp}.png")
    timestamp += 1
    #cv2.imwrite(output_filename, valid)
    fig = plt.figure()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(output)
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

cap.release()
cv2.destroyAllWindows()
