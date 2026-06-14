import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot
from collections import deque
import time
import datetime
import serial
import time
import math
from skimage import measure
from pathlib import Path
import csv
import config
from tacnet_model import TacNet
# from depnet_model import DepNet
# from networks.cnn_model import VisionCNN, AlexNet
from processing import TacImageProcessorRGB
import image_processing_tools as ip
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from resnet import ResNet50
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import re
from utils.visualize import TacThermalSkinVisualize
from utils.simtacls_utils import compute_signed_displacement, create_nodal_radial_vectors
from utils.utils import CameraControl, setup_2dplot
import matplotlib.pyplot as plt
import socket
import os
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(dev)
id_of_cam = 0

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

def load_model_resnet(model_name,num_classes):
    # Load TacNet
    # model_name = 'tacnet_masked_sim_tacballoon_data_12-24-22.pt'
    # model_name = 'tacnet_sim_tacballoon_data_12-20-22.pt'
    MODEL_PATH = config.MODEL_PATH / model_name
    tacnet = ResNet50(num_classes, channels=3)
    print('model [TacNet] was created')
    print('loading the model from {0}'.format(MODEL_PATH))
    tacnet.load_state_dict(torch.load(MODEL_PATH))
    print('---------- Tactile Networks initialized -------------')
    tacnet.to(dev)
    tacnet.eval()

    return tacnet

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
    tacnet.load_state_dict(torch.load(MODEL_PATH))
    print('---------- Tactile Networks initialized -------------')
    tacnet.to(dev)
    tacnet.eval()
    return tacnet

tacnet = load_model_resnet_pre(model_name='shape_real_resnet_pre_tacnet_eletac_02-18-25_00-18_epoch50_lr0.0001.pt',
                    num_classes=6)

DATA_PATH = config.DATA_PATH
DATA_PATH_SHAPE = config.DATA_PATH_SHAPE
INIT_PATH = DATA_PATH_SHAPE / "init_pos.csv"

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

img_folder = '/media/tatu/SSDIOTOUCH2/Tuan/eletac/shape/train/'
img_names = os.listdir(img_folder)

i = 9
img_path = img_folder + img_names[i]
print(img_path)
image = cv2.imread(img_path)
# side, num1, num2, final_num = parse_image_name(img_names[i])
# if side == 'left':
#     y  = num1-5-14.5
# else:
#     y = num1-5+14.5
# x = num2 - 7.5
# d = final_num
# print('ground truth: ', x,y,d)
# truth = np.array([x,y,d])

processed_frame = transform(image)
tac_img = processed_frame.unsqueeze(0)
input_img = tac_img.to(dev)
with torch.no_grad():
    # predict of skin deviation
    estimated_positions_displacement = tacnet(input_img).cpu().numpy()
    print(estimated_positions_displacement)
    print(np.argmax(estimated_positions_displacement))
    # print('distance: ', np.linalg.norm(estimated_positions_displacement-truth))


# cv2.imshow('image',image)
#
# cv2.waitKey(0)
# # and finally destroy/close all open windows
# cv2.destroyAllWindows()