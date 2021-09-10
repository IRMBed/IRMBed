import os
from data.mnistcifar_utils import get_mnist_cifar_env
# from mnistcifar_utils import get_mnist_cifar_env
import pdb
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset

class SpuriousDataset(object):
    def __init__(self, x_array, y_array, env_array, sp_array=None, transform=None):
        assert x_array is not None
        assert y_array is not None
        assert env_array is not None
        self.x_array = x_array
        self.y_array = y_array
        self.env_array = env_array
        self.sp_array = sp_array
        self.transform = transform
        assert len(self.x_array) == len(self.y_array)
        assert len(self.y_array) == len(self.env_array)

    def __len__(self):
        return len(self.x_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        img = self.x_array[idx]
        g = self.env_array[idx]
        if self.sp_array is not None:
            sp = self.sp_array[idx]
        else:
            sp = None
        if self.transform is not None:
            x = self.transform(img)
        else:
            x = img 

        return x,y,g,sp

class CifarMnistSpuriousDataset(Dataset):
    def __init__(self, train_num,test_num,cons_ratios, cifar_classes=(1, 9),train_envs_ratio=None, label_noise_ratio=None, augment_data=True, color_spurious=False, transform_data_to_standard=1, oracle=0):
        self.cons_ratios=cons_ratios
        self.train_num = train_num
        self.test_num = test_num
        self.train_envs_ratio=train_envs_ratio
        self.augment_data = augment_data
        self.oracle = oracle
        self.x_array, self.y_array, self.env_array, self.sp_array= \
            get_mnist_cifar_env(
                train_num=train_num,
                test_num=test_num,
                cons_ratios=cons_ratios,
                train_envs_ratio=train_envs_ratio,
                label_noise_ratio=label_noise_ratio,
                cifar_classes=cifar_classes,
                color_spurious=color_spurious,
                oracle=oracle)
        self.feature_dim = self.x_array.reshape([self.x_array.shape[0], -1]).shape[1]
        self.transform_data_to_standard = transform_data_to_standard
        self.y_array = self.y_array.astype(np.int64)
        self.split_array = self.env_array
        self.n_train_envs = len(self.cons_ratios) - 1
        self.split_dict = {
            "train": range(len(self.cons_ratios) - 1),
            "val": [len(self.cons_ratios) - 1],
            "test": [len(self.cons_ratios) - 1]}
        self.n_classes = 2
        self.train_transform = get_transform_cub(transform_data_to_standard=self.transform_data_to_standard, train=True, augment_data=self.augment_data)
        self.eval_transform = get_transform_cub(transform_data_to_standard=self.transform_data_to_standard, train=False, augment_data=False)

    def return_train_data(self):
        return self.return_data_by_index(self.split_dict["train"])

    def return_test_data(self):
        return self.return_data_by_index(self.split_dict["test"])

    def return_data_by_index(self, env_idx):
        xs = []
        ys = []
        gs = []
        sps = []
        for idx in range(len(self.y_array)):
            if self.split_array[idx] in env_idx:
                x = self.x_array[idx]
                y = self.y_array[idx]
                g = self.env_array[idx]
                sp = self.sp_array[idx]
                xs.append(x)
                ys.append(y)
                gs.append(g)
                sps.append(sp)
        # Figure out split and transform accordingly
        xs = np.stack(xs)
        ys = np.stack(ys)
        gs = np.stack(gs)
        sps = np.stack(sps)
        gs = gs - np.min(gs)
        return xs, ys, gs, sps

def get_provider(batch_size, n_classes, env_nums, train_x=None, train_y=None, train_env=None, train_sp=None, train_transform=None, test_x=None, test_y=None, test_env=None, test_sp=None, test_transform=None):
    class DataProvider(object):
        def __init__(self):
            pass
    train_dataset = SpuriousDataset(
        x_array=train_x,
        y_array=train_y,
        env_array=train_env,
        sp_array=train_sp,
        transform=train_transform)
    test_dataset = SpuriousDataset(
        x_array=test_x,
        y_array=test_y,
        env_array=test_env,
        sp_array=test_sp,
        transform=test_transform)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)
    dp = DataProvider()
    dp.train_loader = train_loader
    dp.test_loader = test_loader
    dp.train_dataset = train_dataset
    dp.test_dataset = test_dataset
    dp.n_classes = n_classes
    dp.env_nums = env_nums
    return dp

def get_transform_cub(transform_data_to_standard, train, augment_data):

    if not transform_data_to_standard:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        scale = 256.0/224.0
        target_resolution = (224, 224)
        assert target_resolution is not None
        if (not train) or (not augment_data):
            # Resizes the image to a slightly larger square then crops the center.
            transform = transforms.Compose([
                transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
                transforms.CenterCrop(target_resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    target_resolution,
                    scale=(0.7, 1.0),
                    ratio=(0.75, 1.3333333333333333),
                    interpolation=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    return transform

if __name__ == "__main__":
    spd, train_loader, val_loader, test_loader, train_data, val_data, test_data = \
    get_data_loader_cifarminst(
        batch_size=100,
        train_num=10000,
        test_num=1800,
        cons_ratios=[0.99,0.8,0.1],
        train_envs_ratio=[0.5,0.5],
        label_noise_ratio=0.1,
        augment_data=False,
        cifar_classes=(2,1),
        color_spurious=1,
        transform_data_to_standard=1)
    # spdc, train_loader, val_loader, test_loader, _, _, _ = get_data_loader_cifarminst(
    #     batch_size=100,
    #     train_num=10000,
    #     test_num=2000,
    #     cons_ratios=[0.99, 0.8, 0.1])
    # print(len(train_loader), len(val_loader), len(test_loader))
    # torch.manual_seed(0)
    loader_iter = iter(train_loader)
    x, y, g = loader_iter.__next__()
    print(y)
    # x, y, g = loader_iter.__next__()
    # print(g)
    # x, y, g = iter(test_loader).__next__()
    # print(g)
    # print(x.shape, y.shape, g.shape)
    # print("y", y)
    # print(g)

    # x, y, g = iter(val_loader).__next__()
    # print(x.shape, y.shape, g.shape)
    # print("y", y)
    # print(g)

    # x, y, g = iter(test_loader).__next__()
    # print(x.shape, y.shape, g.shape)
    # print("y", y)
    # print(g)
