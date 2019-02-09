import model
import loss
import torch
from torch import optim
from torchvision import transforms, datasets
from torch.autograd import Variable
from SiameseNetworkDataset import SiameseNetworkDataset
from Config import Config

train_dataset = datasets.ImageFolder(root=Config.train_dir)
valid_dataset = datasets.ImageFolder(root=Config.valid_dir)
test_dataset = datasets.ImageFolder(root=Config.test_dir)

data_transform = transforms.Compose([
    transforms.Grayscale(),
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
criterion = loss.ContrastiveLoss()
optimizer = optim.Adam(net.parameters(), lr=0.005)

counter = []
loss_history = []
iteration_number = 0

for epoch in range(0, Config.train_number_epochs):
    for i, data in enumerate(train_loader, 0):
        img0, img1, label = data
        img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
        optimizer.zero_grad()
        output1, output2 = net(img0, img1)
        loss_contrastive = criterion(output1, output2, label)
        loss_contrastive.backward()
        optimizer.step()
        if i % 10 == 0:
            print("Epoch number {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
            iteration_number += 10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())
