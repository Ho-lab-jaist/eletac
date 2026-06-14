"""
This scripts train TacNet to reconstruct the skin shape on single-point touch dataset
- Input: Original pair of tactile images (6, 480, 640)
- Ouput: the displacement vectors of free nodes of the artificial skin.
"""


from datetime import datetime, date
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import random_split, ConcatDataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import re
# from simtacls import config
# from simtacls.model.tacnet_model import TacNet
# from simtacls.util.dataset import TactileImageDataset, SingleTactileImageDataset
# from simtacls.util.processing import BinaryTacImageProcessor
# from simtacls.util import image_processing_tools as ip
from torchvision.models import resnet50, ResNet50_Weights
import config
from tacnet_model import TacNet
from resnet import ResNet50
from dataset import TactileImageDataset, SingleTactileImageDataset
from processing import  TacImageProcessorRGB
import image_processing_tools as ip

torch.manual_seed(42)
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print('The hardware device used: {0}'.format(dev))

def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

def get_model(lr, in_nc, num_of_features):
    model = TacNet(in_nc = in_nc, num_of_features = num_of_features)
    # model = ResNet50(in_nc=in_nc, num_of_features=num_of_features)
    model.to(dev)
    # opt = optim.Adam(model.parameters(), lr=lr, betas=(0.04, 0.999))
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    return model, opt

def load_model_train(model_name, lr, in_nc, num_of_features):
    # Extract epoch number from the filename
    match = re.search(r'epoch(\d+)', model_name)
    epoch_number = int(match.group(1)) if match else 0  # Default to 0 if not found
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
    tacnet.train()
    # opt = optim.SGD(tacnet.parameters(), lr=lr, momentum=0.9)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    return tacnet, opt, epoch_number

def load_model_train_resnet(model_name, lr, num_classes):
    # Extract epoch number from the filename
    match = re.search(r'epoch(\d+)', model_name)
    epoch_number = int(match.group(1)) if match else 0  # Default to 0 if not found
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
    tacnet.train()
    opt = optim.SGD(tacnet.parameters(), lr=lr, momentum=0.9)
    return tacnet, opt, epoch_number

def load_model_train_resnet_pre(model_name, lr, num_classes):
    # Extract epoch number from the filename
    match = re.search(r'epoch(\d+)', model_name)
    epoch_number = int(match.group(1)) if match else 0  # Default to 0 if not found
    # Load TacNet
    # model_name = 'tacnet_masked_sim_tacballoon_data_12-24-22.pt'
    # model_name = 'tacnet_sim_tacballoon_data_12-20-22.pt'
    MODEL_PATH = config.MODEL_PATH / model_name
    tacnet = resnet50()
    tacnet.fc = nn.Linear(tacnet.fc.in_features, 3)
    print('model [TacNet] was created')
    print('loading the model from {0}'.format(MODEL_PATH))
    tacnet.load_state_dict(torch.load(MODEL_PATH))
    print('---------- Tactile Networks initialized -------------')
    tacnet.to(dev)
    tacnet.train()
    opt = torch.optim.Adam(tacnet.parameters(), lr=lr)
    return tacnet, opt, epoch_number



def get_model_resnet_pre(lr):
    # model = TacNet(in_nc = in_nc, num_of_features = num_of_features)
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 4) # 3 for location, 4 for location and force
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    return model, opt

def get_model_resnet(lr, num_classes):
    # model = TacNet(in_nc = in_nc, num_of_features = num_of_features)
    model = ResNet50(num_classes, channels=3)
    model.to(dev)
    # opt = optim.Adam(model.parameters(), lr=lr, betas=(0.04, 0.999))
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    return model, opt

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss_position = loss_func(model(xb).float(), yb[:,:4].float()) # 3 for location only, 4 for location and force
    # print('predict: ',model(xb).float(),)
    # print('label: ',yb[:,:4].float())
    # columns_to_keep = [i for i in range(yb.shape[1]) if i % 4 != 3]
    # loss_position = loss_func(model(xb).float()[:, columns_to_keep], yb.float()[:, columns_to_keep])
    # loss_depth = loss_func(0.1*model(xb)[:, 3::4].float(), yb[:, 3::4].float())
    loss = loss_position
    # print("model x: ",torch.max(model(xb)[:, 3::4]))
    # print("y: ",torch.max(yb[:, 3::4]))
    # print("model x all: ", torch.max(model(xb)[:, columns_to_keep]))
    # print("y all: ", torch.max(yb[:, columns_to_keep]))

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def loss_batch_depth(model, loss_func, xb, yb, opt=None):
    # print("model x: ",model(xb).shape)
    # print("y: ",yb.shape)
    # depth_b = yb[:, 3::4]
    # print("model x: ",torch.max(model(xb)[:, 3::4]))
    # print("y: ",torch.max(depth_b))
    loss = loss_func(model(xb)[:, 3::4].float(), yb.float()[:, 3::4])

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl, epoch_number):

    e0 = epoch_number+1

    train_losses = []
    val_losses = []
    training_start_time = datetime.now()
    for epoch in range(epochs):
        epoch_start_time = datetime.now()
        print(
            "Epoch {} started at {}".format(
                epoch + e0, epoch_start_time.strftime("%H:%M:%S")
            )
        )
        model.train()
        running_loss = 0.0 # loss per each batch
        batch = 0
        for xb, yb, _ in train_dl:
            batch+=1
            lossb, numb = loss_batch(model, loss_func, xb, yb, opt=opt)
            # print(numb)
            running_loss += lossb*numb
            if (batch)%50 == 0:
                print('[{}/{}][{}/{}]\tLoss : {}\tElapsed time: {}'
                      .format(epoch+e0,
                              epochs+e0-1,
                              batch, 
                              len(train_dl), 
                              lossb,
                              datetime.now() - epoch_start_time))

        train_loss = running_loss/len(train_ds)
        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb, _ in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print('[{}/{}]\tTraining loss : {}\tValidation loss : {}\tElapsed time: {}'
              .format(epoch+e0,
                      epochs+e0-1,
                      train_loss, 
                      val_loss,
                      datetime.now() - epoch_start_time))

        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
    print(
        "Training finished. Total elapsed time: {}".format(
            datetime.now() - training_start_time
        )
    )
    
    return train_losses, val_losses

def preprocess(x, y):
    return x.to(dev), y.to(dev)

class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            xb, yb = self.func(b['images'], b['displacements'])
            yield (xb, yb, b['image_name'])


# DATA_PATH = config.DATA_PATH / "tac_thermal_arm"
DATA_PATH = config.DATA_PATH
DATA_PATH_REAL = config.DATA_PATH_REAL
INIT_PATH = DATA_PATH_REAL / "init_pos.csv"  # pos2 means with depth, pos1 without depth
# INIT_PATH = DATA_PATH / "node_position" / "node_pos_pre_push_0.01.csv"


# DOUBLE_INPUT_PATH = DATA_PATH / "tac_image" / "double_touch_subset" / "bot"
# DOUBLE_LABEL_PATH = DATA_PATH / "label_pos_double_subset.csv"

# SINGLE_INPUT_PATH = DATA_PATH / "image_and_label" / "image_results"
# SINGLE_LABEL_PATH = DATA_PATH / "label_pos_single.csv"

SINGLE_INPUT_PATH = DATA_PATH_REAL / "train"
SINGLE_LABEL_PATH = DATA_PATH_REAL / "train_label_pos_real.csv"

input_image_size = (256, 256)
# binary_transform = BinaryTacImageProcessor(
#                                             threshold = 140,
#                                             filter_size = 3,
#                                             cropped_size = (400, 400),
#                                             resized_size = input_image_size,
#                                             apply_mask=False,
#                                             mask_radius = 125,
#                                             apply_block_mask = False,
#                                             block_mask_radius = 44,
#                                             block_mask_center = (128,128))
rgb_transform = TacImageProcessorRGB(cropped_size = (250, 250),
                                     resized_size = input_image_size)

# base_transform = BaseTacImageProcessor(cropped_size = (400, 400),
#                                      resized_size = input_image_size)
# Define the desired affine transformation parameters
translation_x = 10
translation_y = 10
# Convert the translation values to ratios
translation_ratio_x = translation_x / input_image_size[1]
translation_ratio_y = translation_y / input_image_size[0]
angle = 5  # Rotation angle in degrees
scale = (0.75, 1.25)  # Scaling factor range
shear = 2  # Shearing angle in degrees
# Apply the custom transformation followed by RandomAffine
transform = transforms.Compose([

    rgb_transform,
    transforms.Lambda(lambda img: F.affine(img, angle=0, translate=(-10, -10), scale=1, shear=0)),
    transforms.RandomAffine(degrees=angle,
                            translate=(translation_ratio_x, translation_ratio_y),
                            scale=scale,
                            shear=shear)

])

single_tactile_ds = SingleTactileImageDataset(
                                       INIT_PATH,
                                       SINGLE_LABEL_PATH,
                                       SINGLE_INPUT_PATH,
                                       input_transform = transform)

tactile_ds = single_tactile_ds
print("The total number of tactile dateset: {}".format(len(single_tactile_ds)))
train_len = int(0.8*len(tactile_ds))
valid_len = len(tactile_ds) - train_len
lengths = [train_len, valid_len]
train_ds, valid_ds = random_split(tactile_ds, lengths)
print("The number of train and valid dateset : {0}-{1}".format(len(train_ds),
                                                               len(valid_ds)))


init_image_path = '/media/tatu/SSDIOTOUCH2/Tuan/eletac/contact/render_init.png'
# init_image = cv2.imread(init_image_path)
# init_processed = transform(init_image)
# print(init_processed.shape)
# print(init_processed[0].shape)
# init_image2 = ip.tensor2img(init_processed[0])
###################### check image #####################
# print(train_ds[56]["image_name"])
# image_sample = train_ds[56]["images"]
idx = 60
print(train_ds[idx]["image_name"])
image_sample = train_ds[idx]["images"]
train_label = train_ds[idx]
print(train_label)
# print(max(image_sample[0][0]))
channel1 = ip.tensor2img(image_sample[0, :, :])
channel2 = ip.tensor2img(image_sample[1, :, :])
channel3 = ip.tensor2img(image_sample[2, :, :])
# print(channel1.shape)
rgb_image = cv2.merge((channel1, channel2, channel3))  # OpenCV uses BGR format by default

# Display the image
cv2.imshow('RGB Image', rgb_image)
#
cv2.waitKey(0)
# and finally destroy/close all open windows
cv2.destroyAllWindows()
###############################################################


#### TRAINING ##########################
# bs = 16 # the number of tactile image paris in one batch
# train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
# train_dl = WrappedDataLoader(train_dl, preprocess)
# valid_dl = WrappedDataLoader(valid_dl, preprocess)
#
# # Check the batch size of the train_dl, valid_dl
# xb, yb, _ = next(iter(train_dl))
# print("Image batch shape : {}".format(xb.shape))
# print("Labels batch shape :  {}".format(yb.shape))
#
# # Get the tactile model
# learning_rate = 1e-4
# # model, opt = get_model(learning_rate,
# #                        in_nc = 3,
# #                        num_of_features = 1)
# epoch_number = 0
# # model, opt, epoch_number = load_model_train_resnet(model_name='real_resnet_pre_tacnet_eletac_02-13-25_00-40_epoch10_lr0.0001.pt',
# #                                                    lr=learning_rate,
# #                                                       num_classes= 3)
#
# # model, opt, epoch_number = load_model_train_resnet_pre(model_name='real_resnet_pre_tacnet_eletac_02-13-25_02-45_epoch150_lr0.0001.pt',
# #                                                    lr=learning_rate,
# #                                                       num_classes= 3)  # 3 for location only, 4 for location and force
# # model, opt = get_model_resnet(learning_rate,4) # 3 for location only, 4 for location and force
#
# # model = resnet50(weights=ResNet50_Weights.DEFAULT)
# # model.fc = nn.Linear(model.fc.in_features, 3)
# # # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # model = model.to(dev)
#
# model, opt = get_model_resnet_pre(lr=learning_rate)
# # model
#
# # Check the weights of the parameters and the number of parameters in the constructed model
# # for name, param in model.named_parameters():
# #     print(f"Layer: {name} | Size: {param.size()} | Num of Element: {param.numel()}")
#
# epochs = 250 # the number of iteration over a training dataset
# # Define the loss function
# mse_loss = nn.L1Loss()
# # mse_loss = nn.MSELoss()
# # opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
# # fit the model
# train_losses, valid_losses = fit(epochs, model, mse_loss, opt, train_dl, valid_dl, epoch_number)
#
# now = datetime.now()
# exp_name = "real_force_resnet_pre_tacnet_eletac_{}_epoch{}_lr{}".format(now.strftime("%m-%d-%y_%H-%M"), epochs+epoch_number, learning_rate)
#
# plt.plot(train_losses, 'r', label='train loss') # plotting t, a separately
# plt.plot(valid_losses, 'b', label='valid loss') # plotting t, b separately
# plt.legend()
# figname = 'training_curve_{}.jpg'.format(exp_name)
# SAVED_TRAINING_CURVE_FILE = config.RESOURCE_PATH / "training_curve/{}".format(figname)
# plt.savefig(SAVED_TRAINING_CURVE_FILE)
# print("Saved Traning Curve Figure")
#
# model_name = '{}.pt'.format(exp_name)
# SAVED_MODEL_FILE = config.MODEL_PATH / model_name
# torch.save(model.state_dict(), SAVED_MODEL_FILE)
# print("Saved Trained Model Weights - {}".format(exp_name))
#
# # Save losses to CSV
# loss_data = pd.DataFrame({
#     "epoch": list(range(1, len(train_losses) + 1)),
#     "train_loss": train_losses,
#     "valid_loss": valid_losses
# })
#
# csv_filename = "losses_{}.csv".format(exp_name)
# SAVED_CSV_FILE = config.RESOURCE_PATH / "training_curve/{}".format(csv_filename)
# loss_data.to_csv(SAVED_CSV_FILE, index=False)
# print("Saved Training Losses to CSV - {}".format(csv_filename))