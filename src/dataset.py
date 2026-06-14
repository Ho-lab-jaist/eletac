"""Tactile Dataset Loader"""

from typing import Callable, Optional
import cv2
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

import config


class TactileImageDataset(Dataset):
    """Class for loading dataset of tactile images"""

    def __init__(
        self,
        init_file,
        label_file,
        img_dir,
        transform_top: Callable[..., Tensor],
        transform_bot: Callable[..., Tensor],
        target_transform: Optional[Callable[..., Tensor]] = None
    ):

        top_img_dir = img_dir / "top/"
        bot_img_dir = img_dir / "bot/"
        self.labels_csv = pd.read_csv(label_file)
        self.labels_csv.set_index("index", inplace=True)
        self.init_pos_csv = pd.read_csv(init_file)
        self.transform_top = transform_top
        self.transform_bot = transform_bot
        self.target_transform = target_transform

        # get image paths
        self.top_img_paths = sorted(top_img_dir.glob("*.png"))
        self.bot_img_paths = sorted(bot_img_dir.glob("*.png"))
        assert len(self.top_img_paths) == len(
            self.bot_img_paths
        ), "Number of images between cameras mismatched"

        # get initial/non-deformed node positions
        self.init_pos = torch.tensor(self.init_pos_csv.iloc[0, 1:], dtype=torch.float)

    def __len__(self):
        return len(self.top_img_paths)

    def __getitem__(self, idx):
        top_img_path = self.top_img_paths[idx]
        bot_img_path = self.bot_img_paths[idx]
        # img_name = top_img_path.name.split(".")[0]
        img_name = top_img_path.name
        # read image in the given directory, and convert the images (.jpg
        # format) into tensors
        image_top = cv2.imread(str(top_img_path))
        image_bot = cv2.imread(str(bot_img_path))
        # get the training labels from the .csv file
        label = (
            torch.tensor(self.labels_csv.loc[img_name], dtype=torch.float)
            - self.init_pos
        )
        image_top = self.transform_top(image_top)
        image_bot = self.transform_bot(image_bot)
        if self.target_transform:
            label = self.target_transform(label)
        # concatenate the two tactile images
        tactile_image = torch.cat((image_top, image_bot), dim=0)
        return {"images": tactile_image, 
                "displacements": label, 
                "image_name": img_name}


class SingleTactileImageDataset(Dataset):
    """Class for loading dataset of tactile images"""

    def __init__(
        self,
        init_file,
        label_file,
        img_dir,
        input_transform: Callable[..., Tensor],
        target_transform: Optional[Callable[..., Tensor]] = None,
    ):

        self.labels_csv = pd.read_csv(label_file)
        self.labels_csv.set_index("index", inplace=True)
        # print(self.labels_csv)
        self.init_pos_csv = pd.read_csv(init_file)
        self.input_transform = input_transform
        self.target_transform = target_transform

        # get image paths
        self.img_paths = sorted(img_dir.glob("*.png"))

        # get initial/non-deformed node positions
        self.init_pos = torch.tensor(self.init_pos_csv.iloc[0, 1:], dtype=torch.float)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_name = img_path.name
        # print(img_name)
        # read image in the given directory, and convert the images (.jpg
        # format) into tensors
        tactile_image = cv2.imread(str(img_path))

        # get the training labels from the .csv file
        label = (
            torch.tensor(self.labels_csv.loc[img_name], dtype=torch.float)
            - self.init_pos
        )

        if self.input_transform:
            tactile_image = self.input_transform(tactile_image)

        if self.target_transform:
            label = self.target_transform(label)

        return {"images": tactile_image, "displacements": label, "image_name": img_name}