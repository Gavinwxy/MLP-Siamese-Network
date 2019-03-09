from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from PIL import Image
import PIL.ImageOps
import torch
import os

class SiameseNetworkDataset(Dataset):

    def __init__(self, imageFolderDataset, transform=None, grayscale=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.grayscale = grayscale

    def __getitem__(self, index):
        root = self.imageFolderDataset.root
        label = random.choice(list(self.imageFolderDataset.class_to_idx.keys()))
        path = os.path.join(root, label)
        img0_name = random.choice(os.listdir(path))
        img0_tuple = os.path.join(path, img0_name), self.imageFolderDataset.class_to_idx[label]
        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            num_imgs = len(os.listdir(path))
            img1_name = random.choice(os.listdir(path))
            while img0_name == img1_name and num_imgs > 1:
                img1_name = random.choice(os.listdir(path))
            img1_tuple = os.path.join(path, img1_name), self.imageFolderDataset.class_to_idx[label]
        else:
            while True:
                # keep looping till a different class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        if self.grayscale:
            img0 = img0.convert("L")
            img1 = img1.convert("L")
        
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)

class TripletDataset(Dataset):

    def __init__(self, imageFolderDataset, transform=None, grayscale=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.grayscale = grayscale

    def __getitem__(self, index):
        root = self.imageFolderDataset.root
        label = random.choice(list(self.imageFolderDataset.class_to_idx.keys()))
        path = os.path.join(root, label)
        img0_name = random.choice(os.listdir(path))
        img0_tuple = os.path.join(path, img0_name), self.imageFolderDataset.class_to_idx[label]

        num_imgs = len(os.listdir(path))
        img1_name = random.choice(os.listdir(path))
        while img0_name == img1_name and num_imgs > 1:
            img1_name = random.choice(os.listdir(path))
        img1_tuple = os.path.join(path, img1_name), self.imageFolderDataset.class_to_idx[label]

        while True:
            # keep looping till a different class image is found
            img2_tuple = random.choice(self.imageFolderDataset.imgs)
            if img0_tuple[1] != img2_tuple[1]:
                break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img2 = Image.open(img2_tuple[0])

        if self.grayscale:
            img0 = img0.convert("L")
            img1 = img1.convert("L")
            img2 = img2.convert("L")

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img0, img1, img2

    def __len__(self):
        return len(self.imageFolderDataset.imgs)