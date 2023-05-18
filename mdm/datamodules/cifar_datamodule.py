import pytorch_lightning as pl
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        test_batch_size: int,
        do_transform: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir

        print("Do Transform in CIFAR 10 Data Module is set to ", do_transform)
        if do_transform:

            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )

        else:

            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )

        self.data_dim = (3, 32, 32)

        self.batch_size = batch_size
        self.test_batch_size = test_batch_size

        self.prepare_data()
        self.setup(stage="fit")
        self.setup(stage="test")

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        self.cifar_train = CIFAR10(
            self.data_dir, train=True, transform=self.transform
        )

        self.cifar_val = CIFAR10(
            self.data_dir, train=False, transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(
            self.cifar_val, batch_size=self.test_batch_size, num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(
            self.cifar_val, batch_size=self.test_batch_size, num_workers=4
        )
