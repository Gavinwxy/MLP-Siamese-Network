from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from PIL import Image
import PIL.ImageOps
import torch
from torchvision import transforms

from Config import Config

class SiameseNetworkDataset(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                # keep looping till a different class image is found

                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)

def data_loaders(model, train_dataset, valid_dataset, test_dataset):
    data_transform = transforms.Compose([
        transforms.Resize(model.input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    siamese_train_dataset = SiameseNetworkDataset(imageFolderDataset=train_dataset,
                                                  transform=data_transform,
                                                  should_invert=False)

    siamese_valid_dataset = SiameseNetworkDataset(imageFolderDataset=valid_dataset,
                                                  transform=data_transform,
                                                  should_invert=False)

    siamese_test_dataset = SiameseNetworkDataset(imageFolderDataset=test_dataset,
                                                 transform=data_transform,
                                                 should_invert=False)

    train_loader = torch.utils.data.DataLoader(siamese_train_dataset, batch_size=Config.train_batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(siamese_valid_dataset, batch_size=Config.valid_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(siamese_test_dataset, batch_size=Config.test_batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader
