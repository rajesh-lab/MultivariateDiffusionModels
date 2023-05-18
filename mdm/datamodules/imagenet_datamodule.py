import os
import torch
import numpy as np
import pytorch_lightning as pl


from PIL import Image
import os.path
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, TensorDataset




"""Create tools for Pytorch to load Downsampled ImageNet (32X32,64X64)

Thanks to the cifar.py provided by Pytorch.

Author: Xu Ma.
Date:   Apr/21/2019

Data Preparation:
    1. Download unsampled data from ImageNet website.
    2. Unzip file  to rootPath. eg: /home/xm0036/Datasets/ImageNet64(no train, val folders)

Remark:
This tool is able to automatic recognize downsampled size.


Use this tool like cifar10 in datsets/torchvision.
"""
class ImageNetDownSample(data.Dataset):


    train_list = [['imagenet_train_jan182023_{}.npz'.format(i)] for i in range(1,11)]
    test_list = [['imagenet_val_jan182023.npz']]

    def __init__(self, root, train=True):
        self.root = os.path.expanduser(root)    
        #self.target_transform = target_transform
        self.train = train  # training set or test set

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, f)
                entry = np.load(file)
                self.train_data.append(entry['data'])
                self.train_labels.append(entry['labels'])
                                    
            self.train_labels = np.concatenate(self.train_labels)
            self.train_data = np.concatenate(self.train_data)
            
            n = self.train_labels.shape[0]                    
            self.train_data = self.train_data.reshape((n, 3, 32, 32)).astype(np.float32) / 255.
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, f)
            entry = np.load(file)
            self.test_data = entry['data']
            
            self.test_labels = entry['labels']        

            # resize label range from [1,1000] to [0,1000),
            # This is required by CrossEntropyLoss
            n = self.test_data.shape[0]
            self.test_data = self.test_data.reshape((n, 3, 32, 32)).astype(np.float32) / 255.        

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]
    
        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)



class ImageNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        test_batch_size: int,
        do_transform: bool = False,
        image_width: int = 32,
            concat_dataset: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.image_width = image_width

        self.data_dim = (3, self.image_width, self.image_width)

        self.batch_size = batch_size
        self.test_batch_size = test_batch_size

        self.concat_dataset = concat_dataset
        self.setup()

    def _verify_splits(self, data_dir: str, split: str) -> None:
        dirs = os.listdir(data_dir)

        if split not in dirs:
            raise FileNotFoundError(
                f"a {split} Imagenet split was not found in {data_dir}, make sure the"
                f" folder contains a subfolder named {split}"
            )

    def setup(
            self, 
    ):

        self.imagenet_train = ImageNetDownSample(root=self.data_dir,
                                train=True)
        self.imagenet_val = ImageNetDownSample(root=self.data_dir,
                                train=False)


    def train_dataloader(self):
        return DataLoader(
            self.imagenet_train, shuffle=True, batch_size=self.batch_size, num_workers=3
        )

    def val_dataloader(self):
        return DataLoader(
            self.imagenet_val, batch_size=self.test_batch_size, num_workers=3
        )

    def test_dataloader(self):
        return DataLoader(
            self.imagenet_val, batch_size=self.test_batch_size, num_workers=3
        )

if __name__ == "__main__":

    data = ImageNetDownSample(root="/scratch/mg3479/imagenet32/", train=True)
    print(data[0])

    dm = ImageNetDataModule(data_dir='./data/imagenet32/',
                            batch_size=32, 
                            test_batch_size=32, 
                            do_transform=False)
    trainloader = dm.train_dataloader()
    valloader = dm.val_dataloader()
    print("Len train", len(trainloader), "batches")
    print("Len val", len(valloader), "batches")


