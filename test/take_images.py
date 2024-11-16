import os
import cv2
import time
import torch

output_dir = "/Users/gr4ntes/PycharmProjects/depth_estimation_anti_spoofing/dataset_real/real"
# Specify the valid directory
os.makedirs(output_dir, exist_ok=True)

# Start video capture from the webcam
cap = cv2.VideoCapture(0)
time.sleep(1)

timestamp = 0

while timestamp < 200:
    ret, frame = cap.read()
    if not ret:
        continue
    cv2.imshow("Frame", frame)
    # Capture the image
    img = frame.copy()

    # Process the image
    output_filename = os.path.join(output_dir, f"output_{timestamp}.png")
    cv2.imwrite(output_filename, img)
    timestamp += 1
    time.sleep(1)

cap.release()
cv2.destroyAllWindows()
