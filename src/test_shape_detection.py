import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot
from collections import deque
import time
import datetime
# import serial
# import time
# import math
# from skimage import measure
from pathlib import Path
import csv
import config
from tacnet_model import TacNet
# from depnet_model import DepNet
# from networks.cnn_model import VisionCNN, AlexNet
from processing import TacImageProcessorRGB
# import image_processing_tools as ip
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
# from resnet import ResNet50
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import re
# from utils.visualize import TacThermalSkinVisualize
# from utils.simtacls_utils import compute_signed_displacement, create_nodal_radial_vectors
# from utils.utils import CameraControl, setup_2dplot
import matplotlib.pyplot as plt
import socket
import os
dev = torch.device("cpu")
print(dev)
id_of_cam = 0
id_of_webcam = 2

cap = cv2.VideoCapture(id_of_cam)
if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

webcam = cv2.VideoCapture(id_of_webcam)
if not webcam.isOpened():
    print("Error: Could not open the camera.")
    exit()

# remove_id
# file0 = '/media/tatu/SSDIOTOUCH2/Tuan/eletac/contact/labels/init_pos.csv'
def parse_image_name(filename):
    # Regular expression to extract "left" or "right", two numbers, and the final number
    match = re.search(r'(left|right)_frame_(\d+)_(\d+)_([\d.]+)\.png', filename)

    if match:
        side = match.group(1)  # "left" or "right"
        num1 = int(match.group(2))  # First number (e.g., 0)
        num2 = int(match.group(3))  # Second number (e.g., 0)
        final_number = float(match.group(4))  # Last number (e.g., 5.0)
        return side, num1, num2, final_number
    else:
        return None, None, None, None  # Return None if pattern doesn't match

def load_model_resnet_pre(model_name, num_classes):
    # Extract epoch number from the filename
    # Load TacNet
    # model_name = 'tacnet_masked_sim_tacballoon_data_12-24-22.pt'
    # model_name = 'tacnet_sim_tacballoon_data_12-20-22.pt'
    MODEL_PATH = config.MODEL_PATH / model_name
    tacnet = resnet50()
    tacnet.fc = nn.Linear(tacnet.fc.in_features, 6)
    print('model [TacNet] was created')
    print('loading the model from {0}'.format(MODEL_PATH))
    tacnet.load_state_dict(torch.load(MODEL_PATH,  map_location=torch.device('cpu')))
    print('---------- Tactile Networks initialized -------------')
    tacnet.to(dev)
    tacnet.eval()
    return tacnet

# tacnet = load_model_resnet_pre(model_name='shape_real_resnet_pre_tacnet_eletac_03-12-25_10-30_epoch100_lr0.0001.pt',
#                     num_classes=6)

tacnet = load_model_resnet_pre(model_name='shape_real_resnet_pre_tacnet_eletac_02-18-25_00-18_epoch50_lr0.0001.pt',
                    num_classes=6)

DATA_PATH = config.DATA_PATH
DATA_PATH_REAL = config.DATA_PATH_REAL
INIT_PATH = DATA_PATH_REAL / "init_pos.csv"

# INIT_PATH = DATA_PATH / "labels/init_pos.csv"
init_pos_csv = pd.read_csv(INIT_PATH)
init_positions = (np.array(init_pos_csv.iloc[0, 1:4], dtype=float)
                  .reshape(-1, 3))
input_image_size = (256, 256)
rgb_transform = TacImageProcessorRGB(cropped_size = (400, 400),
                                     resized_size = input_image_size)
transform = transforms.Compose([

    rgb_transform,
    transforms.Lambda(lambda img: F.affine(img, angle=0, translate=(-10, -10), scale=1, shear=0)),


])

# img_folder = 'D:/holab_local/off_campus/code/dobot_code/dobot_control/data/images_force/test/'
# img_names = os.listdir(img_folder)
shapes = ['circle', 'circle_hollow', 'dot', 'square', 'stripe', 'triangle']

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if the camera feed is unavailable
    ret2, frame2 = webcam.read()
    cv2.imshow("gripper", frame2)

    with torch.no_grad():
        # Predict skin deviation
        processed_frame = transform(frame)
        tac_img = processed_frame.unsqueeze(0)
        input_img = tac_img.to(dev)
        estimated_positions_displacement = tacnet(input_img).cpu().numpy()
        shape_id = np.argmax(estimated_positions_displacement[0])

        predicted_shape = "None"
        confidence = estimated_positions_displacement[0][shape_id]

        if confidence > 1:
            predicted_shape = shapes[shape_id]
            print('Predicted percentage:', confidence)
            print('Predicted shape:', predicted_shape)

        # Overlay predicted shape on the frame
        text = f"Shape: {predicted_shape} ({confidence:.2f})"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame with the prediction
    cv2.imshow('Camera Feed', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# while True:
#     ret, frame = cap.read()
#     cv2.imshow('Camera Feed', frame)
#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#     with torch.no_grad():
#         # predict of skin deviation
#         processed_frame = transform(frame)
#         tac_img = processed_frame.unsqueeze(0)
#         input_img = tac_img.to(dev)
#         estimated_positions_displacement = tacnet(input_img).cpu().numpy()
#         shape_id = np.argmax(estimated_positions_displacement[0])
#         # print(estimated_positions_displacement[0])
#         # print(shape_id)
#         if estimated_positions_displacement[0][shape_id] >1:
#             # if shapes[shape_id] != 'stripe' or estimated_positions_displacement[0][shape_id] > 6:
#             print('Predicted percentage: ', estimated_positions_displacement[0][shape_id])
#             print('Predicted shape: ', shapes[shape_id])
#         print(' ')