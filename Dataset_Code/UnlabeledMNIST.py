import pytorch_lightning as pl
import torch
import numpy as np
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader, random_split

# TODO: Create 2 sets of images from MNIST-train: Labeled 200, Unlabeled the rest
    # Labeled train
    # Unlabeled train
# TODO: Train ENCODER-DECODER architecture using error as squared distance between input and output pixels
# TODO: Freeze ENCODER and train a classifier with the 100 labeled images
# TODO: Validation set (100 images) > 50% accuracy
# TODO: Test set (100 images) > 50% accuracy


class UnlabeledMNIST(pl.LightningDataModule):
    def __init__(self, batch_size=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = 4
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        self.mnist_train, self.mnist_test = None, None
        self.labeled_train, self.labeled_val, self.unlabeled = None, None, None

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        self.mnist_train = torchvision.datasets.MNIST(root='../datasets', train=True,
                                                      download=True, transform=self.transform)

        self.mnist_test = torchvision.datasets.MNIST(root='../datasets', train=False,
                                                     download=True, transform=self.transform)

    def setup(self):
        # make assignments here (val/train/test split)
        # called on every process in DDP

        train_dataset_size = len(self.mnist_train)
        self.labeled_train, self.labeled_val, self.unlabeled = random_split(
            self.mnist_train, [100, 100, train_dataset_size - 200])

    def labeled_train_dataloader(self):
        return DataLoader(self.labeled_train, batch_size=self.batch_size, num_workers=4)

    def labeled_val_dataloader(self):
        return DataLoader(self.labeled_val, batch_size=self.batch_size, num_workers=4)

    def unlabeled_dataloader(self):
        return DataLoader(self.unlabeled, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=4)
