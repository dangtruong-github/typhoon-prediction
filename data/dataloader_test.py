from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import pytorch_lightning as pl

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./datasets', batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def prepare_data(self):
        datasets.CIFAR10(root=self.data_dir, train=True, download=True)
        datasets.CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        cifar_full = datasets.CIFAR10(root=self.data_dir, train=True, transform=self.transform)
        self.train_set, self.val_set = random_split(cifar_full, [45000, 5000])
        self.test_set = datasets.CIFAR10(root=self.data_dir, train=False, transform=self.transform)

class CIFAR10Loader(CIFAR10DataModule):
    def __init__(self, data_dir='./datasets', batch_size=64):
        super().__init__(data_dir, batch_size)       

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)
