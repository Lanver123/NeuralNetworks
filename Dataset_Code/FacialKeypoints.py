import os
import torch
import pandas as pd
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from zipfile import ZipFile
from utils import ROOT_DIRECTORY

class FacialKeypointsDataset(Dataset):
    def __init__(self, root_dir: str, file_csv: str):
        """
        The data has the following shape:
        6451 rows x 31 columns
            Columns: First 30 columns: (x,y) coordinates of 15 facial keypoints
                     Last column: 1x96x96 pixel values of image
        """
        with ZipFile('{}/facial_keypoints.zip'.format(root_dir)) as zip:
            with zip.open(file_csv) as myZip:
                self.keypoints_df = pd.read_csv(myZip, index_col=0)

    def __len__(self):
        return len(self.keypoints_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'keypoints': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

class FacialKeypointsDataloader(pl.LightningDataModule):
    def __init__(self, batch_size=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = 4
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        self.train = None
        self.val = None

    def prepare_data(self):
        # download, split, etc...
        self.train = FacialKeypointsDataset(root_dir='{}/Datasets'.format(ROOT_DIRECTORY), file_csv='training.csv')
        self.val = FacialKeypointsDataset(root_dir='{}/Datasets'.format(ROOT_DIRECTORY), file_csv='val.csv')

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=4)
