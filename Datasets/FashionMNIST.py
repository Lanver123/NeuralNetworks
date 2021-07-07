import pytorch_lightning as pl
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split


class FashionMNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = 4
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        self.fashion_mnist_train_val, self.fashion_mnist_test = None, None
        self.train_dataset, self.val_dataset = None, None

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        self.fashion_mnist_train_val = torchvision.datasets.FashionMNIST(root='../datasets', train=True,
                                                                         download=True, transform=self.transform)

        self.fashion_mnist_test = torchvision.datasets.FashionMNIST(root='../datasets', train=False,
                                                                    download=True, transform=self.transform)

    def setup(self):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        self.train_dataset, self.val_dataset = random_split(
            self.fashion_mnist_train_val, [50000, 10000])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.fashion_mnist_test, batch_size=self.batch_size)
