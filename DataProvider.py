import os
import torch
from torchvision import transforms, datasets
from torch.autograd import Variable

root = 'D:/Lab/mlpproject/Extracted_Base'

data_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(root=os.path.join(root, 'train'),
                                     transform=data_transform)

valid_dataset = datasets.ImageFolder(root=os.path.join(root, 'valid'),
                                     transform=data_transform)

test_dataset = datasets.ImageFolder(root=os.path.join(root, 'test'),
                                    transform=data_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=True)

use_gpu = torch.cuda.is_available()

for loader in [train_loader, valid_loader, test_loader]:
    for data in loader:
        inputs, labels = data

        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
