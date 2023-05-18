import pytorch_lightning as pl
import numpy as np

import joblib
import os

import torchvision.transforms as transforms
from PIL import Image
import torch 
from torch.utils.data import DataLoader


class LMDBDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform):                
        self.transform = transform
        self.path = path


    def __getitem__(self, index):
        img = joblib.load(os.path.join(self.path, f"{index}.npy"))
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)

        return img, 0

    def __len__(self):
        return 27000 if "train" in self.path else 3000


class CelebA256DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        test_batch_size: int,
    ):
        super().__init__()
        self.data_dir = data_dir

        c, h, w = (3, 256, 256)

        self.train_transform = transforms.Compose([
            transforms.Resize(h),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        self.valid_transform = transforms.Compose([
        transforms.Resize(h),
        transforms.ToTensor(),
        ])

        self.data_dim = (c, h, w)

        self.batch_size = batch_size
        self.test_batch_size = test_batch_size

        self.setup(stage="fit")
        self.setup(stage="test")


    def setup(self, stage: str):
        if stage == "fit":
            self.celeba_train = LMDBDataset(path=os.path.join(self.data_dir, "celeba/celeba-joblib/train"), transform=self.train_transform)
            
        elif stage == "test":
            self.celeba_val = LMDBDataset(path=os.path.join(self.data_dir, "celeba/celeba-joblib/validation"), transform=self.train_transform)
        else:
            raise NotImplementedError(f"State {stage} not implemented")

    def train_dataloader(self):
        return DataLoader(self.celeba_train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.celeba_train, batch_size=self.test_batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.celeba_train, batch_size=self.test_batch_size, num_workers=4)
