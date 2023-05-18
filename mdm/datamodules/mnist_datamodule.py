import pytorch_lightning as pl
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir: str, batch_size: int, test_batch_size: int,
    ):
        super().__init__()
        self.data_dir = data_dir

        self.transform = transforms.Compose(
            [transforms.ToTensor()]
        )
        
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size

        self.data_dim = (28, 28)

        self.prepare_data()
        self.setup(stage="fit")
        self.setup(stage="test")

    def prepare_data(self) -> None:
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str) -> None:

        if stage == "fit":
            self.mnist_train = MNIST(
                self.data_dir, train=True, transform=self.transform
            )            
            
        elif stage == "test":
            self.mnist_val = MNIST(
                self.data_dir, train=False, transform=self.transform
            )
        else:
            raise NotImplementedError(f"State {stage} not implemented")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_val, batch_size=self.test_batch_size, num_workers=4)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_val, batch_size=self.test_batch_size, num_workers=4)

