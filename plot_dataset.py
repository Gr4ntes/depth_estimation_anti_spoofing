import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# Set the paths for the two folders containing images
folder1_path = 'dataset/test/fake'
folder2_path = 'dataset/test/valid'

# Find and sort image paths in each folder
folder1_images = sorted(glob.glob(os.path.join(folder1_path, '*.png')))[:8]
folder2_images = sorted(glob.glob(os.path.join(folder2_path, '*.png')))[:8]

# Load the images
images1 = [plt.imread(image_path) for image_path in folder1_images]
images2 = [plt.imread(image_path) for image_path in folder2_images]

# Create an 8x8 grid
fig, axes = plt.subplots(4, 4, figsize=(4, 4))

# Flatten axes to enable iteration
axes = axes.flatten()

# Plot images
for i, ax in enumerate(axes):
    if i < 8:
        ax.imshow(images1[i])
    else:
        ax.imshow(images2[i-8])
    ax.axis('off')

# Display the figure
plt.show()