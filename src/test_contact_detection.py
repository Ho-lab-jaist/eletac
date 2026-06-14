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

def load_model(model_name, in_nc, num_of_features):
    # Load TacNet
    # model_name = 'tacnet_masked_sim_tacballoon_data_12-24-22.pt'
    # model_name = 'tacnet_sim_tacballoon_data_12-20-22.pt'
    MODEL_PATH = config.MODEL_PATH / model_name
    tacnet = TacNet(in_nc = in_nc,
                    num_of_features = num_of_features)
    print('model [TacNet] was created')
    print('loading the model from {0}'.format(MODEL_PATH))
    tacnet.load_state_dict(torch.load(MODEL_PATH))
    print('---------- Tactile Networks initialized -------------')
    tacnet.to(dev)
    tacnet.eval()

    return tacnet

# def load_model_resnet(model_name,num_classes):
#     # Load TacNet
#     # model_name = 'tacnet_masked_sim_tacballoon_data_12-24-22.pt'
#     # model_name = 'tacnet_sim_tacballoon_data_12-20-22.pt'
#     MODEL_PATH = config.MODEL_PATH / model_name
#     tacnet = ResNet50(num_classes, channels=3)
#     print('model [TacNet] was created')
#     print('loading the model from {0}'.format(MODEL_PATH))
#     tacnet.load_state_dict(torch.load(MODEL_PATH))
#     print('---------- Tactile Networks initialized -------------')
#     tacnet.to(dev)
#     tacnet.eval()

    return tacnet

def load_model_resnet_pre(model_name, num_classes):
    # Extract epoch number from the filename
    # Load TacNet
    # model_name = 'tacnet_masked_sim_tacballoon_data_12-24-22.pt'
    # model_name = 'tacnet_sim_tacballoon_data_12-20-22.pt'
    MODEL_PATH = config.MODEL_PATH / model_name
    tacnet = resnet50()
    tacnet.fc = nn.Linear(tacnet.fc.in_features, 3)
    print('model [TacNet] was created')
    print('loading the model from {0}'.format(MODEL_PATH))
    tacnet.load_state_dict(torch.load(MODEL_PATH,  map_location=torch.device('cpu')))
    print('---------- Tactile Networks initialized -------------')
    tacnet.to(dev)
    tacnet.eval()
    return tacnet

tacnet = load_model_resnet_pre(model_name='real_resnet_pre_tacnet_eletac_02-13-25_04-01_epoch250_lr0.0001.pt',
                    num_classes=3)

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
#
# i = 100
# img_path = img_folder + img_names[i]
# print(img_path)
# image = cv2.imread(img_path)
# side, num1, num2, final_num = parse_image_name(img_names[i])
# if side == 'left':
#     y  = num1-5-14.5
# else:
#     y = num1-5+14.5
# x = num2 - 7.5
# d = final_num
# # print('ground truth: ', x,y,d)
# truth = np.array([x,y,d])
#
# processed_frame = transform(image)
# tac_img = processed_frame.unsqueeze(0)
# input_img = tac_img.to(dev)
content = []
index = []
pos = []
file = '/media/tatu/SSDIOTOUCH2/Tuan/eletac/force_sim/init.csv'
with open(file, 'r') as file:
    csvreader = csv.reader(file)
    for row in csvreader:
        content.append(row)

for i in range(0, len(content)):
    index.extend([int(content[i][1])])
    pos.append([float(content[i][2].replace('np.float64(', '').replace(')', '').strip()),
                float(content[i][3].replace('np.float64(', '').replace(')', '').strip()),
                float(content[i][4].replace('np.float64(', '').replace(')', '').strip())])

x0 = [col[0] for col in pos]
y0 = [col[2] for col in pos]
z0 = [col[1] for col in pos]
# Create a figure and 3D axis for live plotting
fig, ax = plt.subplots()
plt.ion()  # Enable interactive mode for real-time updating
while True:
    ret, frame = cap.read()
    ret2, frame2 = webcam.read()
    # cv2.imshow('Camera Feed', frame)
    cv2.imshow('input_image', frame)
    cv2.imshow("gripper", frame2)
    with (torch.no_grad()):
        # predict of skin deviation
        processed_frame = transform(frame)
        tac_img = processed_frame.unsqueeze(0)
        input_img = tac_img.to(dev)
        estimated_positions_displacement = tacnet(input_img).cpu().numpy().reshape(-1, 3)

        # print('distance: ', np.linalg.norm(estimated_positions_displacement))
        # Clear previous plot and re-plot
        thresh = 8
        if estimated_positions_displacement[0][2] > 3.5 and not(-thresh < -estimated_positions_displacement[0][1] < thresh and -thresh < estimated_positions_displacement[0][0] < thresh):
        # if estimated_positions_displacement[0][2] > 3:
            print(estimated_positions_displacement[0])
            x_pred = -estimated_positions_displacement[0][1]
            y_pred = estimated_positions_displacement[0][0]
            z_pred = 44.82 - 0*estimated_positions_displacement[0][2]
            size = 100
            print(x_pred, y_pred)
        else:
            x_pred = 0
            y_pred = 0
            z_pred = 0
            size = 0.1
        ax.clear()
        ax.scatter(x0, y0, c='green', s=10,)  # Existing points
        ax.scatter(x_pred, y_pred, c='r', marker='o', s=size, label="Estimated Contact Position")  # Prediction


        # Labels and limits
        ax.set_xlabel('Y')
        ax.set_ylabel('X')
        # ax.set_zlabel('Z')
        ax.legend()

        # Update plot
        plt.draw()
        plt.pause(0.01)  # Pause to allow updating
        if cv2.waitKey(1) & 0xFF == ord('q'):
            plt.close()
            break

# cv2.imshow('image',image)
#
# cv2.waitKey(0)
cap.release()
webcam.release()
# and finally destroy/close all open windows
cv2.destroyAllWindows()