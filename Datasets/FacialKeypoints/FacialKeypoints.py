import os
import torch
import pandas as pd
import numpy as np
from skimage import io
from torch.utils.data import Dataset, DataLoader, random_split
from zipfile import ZipFile
import Datasets.FacialKeypoints.FacialKeypoints
import os.path

class FacialKeypointsDataset(Dataset):
    def __init__(self, file_csv: str):
        """
        The data has the following shape:
        6451 rows x 31 columns
            Columns: First 30 columns: (x,y) coordinates of 15 facial keypoints
                     Last column: 1x96x96 pixel values of image
        """
        print(os.path.abspath(Datasets.__file__))

        with ZipFile('{}/facial_keypoints.zip'.format()) as zip:
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

FacialKeypointsDataset(file_csv='training.csv')

class FacialKeypointsDataloader(pl.LightningDataModule):
    def __init__(self, batch_size=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = 4
        self.train_val, self.train, self.val = None
        self.test = None

    def prepare_data(self):
        # download, split, etc...
        self.train_val = FacialKeypointsDataset(root_dir='{}/Datasets'.format(ROOT_DIRECTORY), file_csv='training.csv')
        train_val_size = len(self.train_val)
        train_size = floor(train_val_size*0.75)
        val_size = train_val_size - train_size

        self.test = FacialKeypointsDataset(root_dir='{}/Datasets'.format(ROOT_DIRECTORY), file_csv='val.csv')
        self.train, self.val = random_split(self.fashion_mnist_train_val, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=4)
