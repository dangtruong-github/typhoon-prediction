import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split

from torchvision import datasets, transforms
import pytorch_lightning as pl

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./datasets', batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        # Define the transformation
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.setup()

    def prepare_data(self):
        datasets.MNIST(root='./datasets', train=True, download=True, transform=self.transform)
        datasets.MNIST(root='./datasets', train=False, download=True, transform=self.transform)

    def __filter_digits(self, dataset, digits=[0, 1]):
        indices = [i for i, (x, y) in enumerate(dataset) if y in digits]
        return Subset(dataset, indices)

    def setup(self, stage=None):
        self.prepare_data()

        mnist_full = datasets.MNIST(root=self.data_dir, train=True,
                                    transform=self.transform)
        mnist_full_samples = len(mnist_full)
        mnist_train = int(mnist_full_samples * 0.9)

        self.train_set, self.val_set = random_split(
            mnist_full,
            [mnist_train, mnist_full_samples - mnist_train]
        )
        self.test_set = datasets.MNIST(root=self.data_dir, train=False,
                                         transform=self.transform)

        self.train_set = self.__filter_digits(self.train_set)
        self.val_set = self.__filter_digits(self.val_set)
        self.test_set = self.__filter_digits(self.test_set)

class MNISTLoader(MNISTDataModule):
    def __init__(self, data_dir='./datasets', batch_size=64):
        super().__init__(data_dir, batch_size)       

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)
