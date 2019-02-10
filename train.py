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


train_dataset = datasets.ImageFolder(root=Config.train_dir)
valid_dataset = datasets.ImageFolder(root=Config.valid_dir)
test_dataset = datasets.ImageFolder(root=Config.test_dir)

data_transform = transforms.Compose([
    transforms.Resize(Config.size),
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

net = model.DeepID().cuda()
#net = model.DeepID()
criterion = loss.ContrastiveLoss()
optimizer = optim.Adam(net.parameters(), lr=0.005)

counter = []
iteration_number = 0
loss_per_epoch = {"train_loss": [], "valid_loss": []}
num_batch_train = len(train_loader)
num_batch_val = len(valid_loader)

for epoch in range(Config.train_number_epochs):
    loss_per_batch = {"train_loss": [], "valid_loss": []}
    print("Epoch number: %d" % (epoch+1))

    with tqdm.tqdm(total=num_batch_train) as pbar_train:
        for i, data in enumerate(train_loader, 0): # Train for one epoch
            img0, img1, label = data
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            #img0, img1, label = img0, img1, label
            optimizer.zero_grad()
            output1, output2 = net(img0, img1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
            loss = loss_contrastive.item()
            loss_per_batch["train_loss"].append(loss)
            pbar_train.set_description("train loss: {:.4f}".format(np.mean(loss_per_batch['train_loss'])))
            pbar_train.update(1)

    with tqdm.tqdm(total=num_batch_val) as pbar_val:
        for i, data in enumerate(valid_loader, 0): # Evaluation for one epoch
            img0, img1, label = data
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
            #img0, img1, label = img0, img1, label
            output1, output2 = net.forward(img0, img1)
            loss_contrastive = criterion(output1, output2, label)
            loss = loss_contrastive.item()
            loss_per_batch["valid_loss"].append(loss)
            pbar_val.set_description("valid loss: {:.4f}".format(np.mean(loss_per_batch['valid_loss'])))
            pbar_val.update(1)
    
    print()
    for key, value in loss_per_batch.items(): # Collect average loss for current epoch
        loss_per_epoch[key].append(np.mean(value))


torch.save(net.state_dict(), f="/Users/yantiz/Desktop/ML课程/MLP/MLP-Siamese-Network/model/model.pth") # Model saving, Only save the parameters (Recommended)

model = model.DeepID().cuda()
#model = model.DeepID()
model.load_state_dict(torch.load("/Users/yantiz/Desktop/ML课程/MLP/MLP-Siamese-Network/model/model.pth")) # Instantialize the model before loading the parameters
