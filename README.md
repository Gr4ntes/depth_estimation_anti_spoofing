# Monocular Depth Estimation for Face Anti-Spoofing

---

This repository contains an anti-fraud system that can be used to enhance performance in face recognition tasks.
It uses **depth estimation** to prevent attacks when criminals try to use picture of a person registered in a system to fool the face recognition model and get access to valuable information.

The process involves first estimation the depth of the image taken for face recognition with MiDaS monocular depth estimation model,
then extracting HoG values from the image and using them as input for the Random Forest Classifier.

## Usage

The python packages can be downloaded from the *requirements.txt* file.
The *main.py* python script launches the demo program. 

## Testing
The system was tested and compared with other solutions including LBP-based classifier, CNN, ViT. The designed system managed to achieve respectable results, placing second in the comparison behind CNN.