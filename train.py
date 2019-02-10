import os
import model
import loss
import torch
from torch import optim
from torchvision import transforms, datasets
from torch.autograd import Variable
from dataset import SiameseNetworkDataset
from Config import Config
import tqdm
import numpy as np


def train(model, loss_func, lr):
    #net = type(model)().cuda()
    net = type(model)()
    criterion = loss_func
    optimizer = optim.Adam(net.parameters(), lr=lr)

    counter = []
    iteration_number = 0
    loss_per_epoch = {"train_loss": [], "valid_loss": []}
    num_batch_train = len(train_loader)
    num_batch_val = len(valid_loader)

    for epoch in range(Config.train_number_epochs):
        loss_per_batch = {"train_loss": [], "valid_loss": []}
        print("Epoch number: %d" % (epoch+1))

        with tqdm.tqdm(total=num_batch_train) as pbar_train:
            for i, data in enumerate(train_loader, 0): # Train for one batch
                img0, img1, label = data
                #img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
                img0, img1, label = img0, img1, label
                optimizer.zero_grad()
                output1, output2 = net(img0, img1)
                loss = criterion(output1, output2, label)
                loss.backward()
                optimizer.step()
                #loss = loss_func.item()
                loss_per_batch["train_loss"].append(loss.item())
                pbar_train.set_description("train loss: {:.4f}".format(np.mean(loss_per_batch['train_loss'])))
                pbar_train.update(1)

        with tqdm.tqdm(total=num_batch_val) as pbar_val:
            for i, data in enumerate(valid_loader, 0): # Evaluation for one batch
                img0, img1, label = data
                #img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
                img0, img1, label = img0, img1, label
                output1, output2 = net.forward(img0, img1)
                loss = criterion(output1, output2, label)
                #loss = loss.item()
                loss_per_batch["valid_loss"].append(loss.item())
                pbar_val.set_description("valid loss: {:.4f}".format(np.mean(loss_per_batch['valid_loss'])))
                pbar_val.update(1)
        
        for key, value in loss_per_batch.items(): # Collect average loss for current epoch
            loss_per_epoch[key].append(np.mean(value))

    torch.save(net.state_dict(), f=os.path.join(Config.saved_models_dir, 'model' + str(search_times) + 'pth')) # Model saving, Only save the parameters (Recommended)

    #net = type(model)().cuda()
    net = type(model)()
    net.load_state_dict(torch.load(os.path.join(Config.saved_models_dir, 'model' + str(search_times) + 'pth'))) # Instantialize the model before loading the parameters

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


train_dataset = datasets.ImageFolder(root=Config.train_dir)
valid_dataset = datasets.ImageFolder(root=Config.valid_dir)
test_dataset = datasets.ImageFolder(root=Config.test_dir)

grid_search = {
    "model": [model.DeepID(), model.ChopraNet()],
    "loss_func": [loss.ContrastiveLoss()],
    "lr": [0.005]
}

search_times = 1
for model in grid_search['model']:
    train_loader, valid_loader, test_loader = data_loaders(model, train_dataset, valid_dataset, test_dataset)

    for loss_func in grid_search['loss_func']:
        for lr in grid_search['lr']:
            train(model, loss_func, lr)

            print("\nGrid search {} is completed.\n".format(search_times))
            search_times += 1
