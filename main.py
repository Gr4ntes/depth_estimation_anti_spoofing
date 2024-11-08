import subprocess
import sys
import os
import cv2
import pickle
import torch
from datetime import datetime
from matplotlib import pyplot as plt
import face_recognition
import numpy as np
from skimage.feature import hog
from PySide6 import QtCore, QtWidgets, QtGui
import time
import utils

model_dict = pickle.load(open('model.p', 'rb'))
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

def create_MessageBox(type, title, text):
    message_box = QtWidgets.QMessageBox()
    message_box.setWindowTitle(title)
    message_box.setText(text)
    message_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
    if type == 'error':
        message_box.setIcon(QtWidgets.QMessageBox.Critical)
    message_box.exec()

class CamRunnable(QtCore.QRunnable):
    def __init__(self, changePixmap):
        super().__init__()
        self._changePixmap = changePixmap
        self._run_flag = True
        self.cap = None

    def run(self):
        while self._run_flag:
            self.cap = cv2.VideoCapture(0)
            ret, frame = self.cap.read()
            self.recent_frame = frame
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytesPerLine = ch * w
                convertToQtFormat = QtGui.QImage(rgb_image.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
                self._changePixmap.emit(convertToQtFormat)
        self.cap.release()


class CamThread(QtCore.QObject):
    changePixmap = QtCore.Signal(QtGui.QImage)

    def __init__(self):
        super().__init__()
        self.runnable = CamRunnable(self.changePixmap)
        self.thread_pool = QtCore.QThreadPool.globalInstance()
        self.thread_pool.start(self.runnable)

    def cleanup(self):
        self.runnable._run_flag = False
        self.thread_pool.waitForDone()


class App(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()

        self.login_button = QtWidgets.QPushButton("Login")
        self.register_button = QtWidgets.QPushButton("Register")
        self.images_checkbox = QtWidgets.QCheckBox()
        self.images_checkbox.setChecked(True)
        self.label = QtWidgets.QLabel(self)
        self.th = CamThread()

        self.login_button.clicked.connect(self.login)
        self.register_button.clicked.connect(self.register)

        self.init_ui()
        self.db_dir = './db'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)

    @QtCore.Slot(QtGui.QImage)
    def set_image(self, image):
        cropped_image = image.copy(image.width() * 0.3, image.height() * 0.2, image.width() * 0.4,
                                   image.height() * 0.6)
        cropped_pixmap = QtGui.QPixmap.fromImage(cropped_image)

        mirror_transform = QtGui.QTransform()
        mirror_transform.scale(-1, 1)
        self.label.setPixmap(cropped_pixmap.transformed(mirror_transform))

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        row_layout = QtWidgets.QHBoxLayout(self)

        header_text = QtWidgets.QLabel("Face Recognition + Anti-Spoofing", self)
        font = QtGui.QFont("Helvetica", 20)
        font.setBold(True)
        header_text.setFont(font)
        row_layout.addWidget(header_text)
        row_layout.addStretch()
        checkbox_text = QtWidgets.QLabel("Save images", self)
        checkbox_font = QtGui.QFont("Helvetica", 12)
        checkbox_text.setFont(checkbox_font)
        row_layout.addWidget(checkbox_text)
        row_layout.addWidget(self.images_checkbox)
        layout.addLayout(row_layout)
        layout.addWidget(self.label)
        layout.addWidget(self.login_button)
        layout.addWidget(self.register_button)

        self.th.changePixmap.connect(self.set_image)
        self.show()

    def register(self):
        cv2.imwrite(os.path.join(self.db_dir, '{}.jpg'.format(datetime.now())), self.th.runnable.recent_frame)

    def login(self):
        original_img_path = './original_img.png'
        depth_img_path = "./depth_img.png"
        hog_img_path = "./hog_img.png"

        cv2.imwrite(original_img_path, self.th.runnable.recent_frame)

        start_time = time.perf_counter()
        result = utils.identify_face(self.db_dir, original_img_path)
        if not result:
            create_MessageBox('error', 'Error',
                              'Either a user is not registered or no faces detected. \nPlease try again later.')
        else:
            start_spoofing_time = time.perf_counter()
            predicted_result = utils.validate(original_img_path, depth_img_path, hog_img_path, self.images_checkbox.isChecked())
            end_time = time.perf_counter()
            spoofing_elapsed = end_time - start_spoofing_time
            print(f"Anti-Spoofing time: {spoofing_elapsed:.4f} seconds")
            elapsed = end_time - start_time
            print(f"Total time: {elapsed:.4f} seconds")
            if predicted_result == "valid":
                create_MessageBox('info', 'Success!', "Login attempt was successful, Welcome back!")
            else:
                create_MessageBox('error', 'Error', "It looks like you are fake!")
        if not self.images_checkbox.isChecked():
            os.remove(original_img_path)

    def cleanup(self):
        self.th.cleanup()
        self.close()

    def closeEvent(self, event):
        self.cleanup()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    widget = App()

    sys.exit(app.exec())
