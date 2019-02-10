import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class DeepID(nn.Module):
    input_size = (39, 31)

    def __init__(self):
        super(DeepID, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(20, 40, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(40, 60, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Conv2d(60, 80, kernel_size=2)
        self.fc = nn.Linear(520, 160)

    def forward_once(self, x):
        ### Locally connected layers needed here !!!
        out1 = self.conv1(x)
        out2 = F.relu(self.conv2(out1))
        out1 = out1.view(out1.shape[0], -1)
        out2 = out2.view(out2.shape[0], -1)
        out3 = torch.cat((out1, out2), 1)
        out3 = self.fc(out3)
        return out3

    def forward(self, input1, input2):
        output1 = self.forward_once(input1) 
        output2 = self.forward_once(input2)
        return output1, output2


class ChopraNet(nn.Module):
    input_size = (56, 46)

    def __init__(self):
        super(ChopraNet, self).__init__()
        # Convolution layer with 15 7*7 kernels
        self.conv1 = nn.Conv2d(1, 15, 7)
        # Subsampling layer (an average pooling layer
        # & multiply a trainable coefficient, add a trainable bias & go through sigmoid)
        self.avgpool1 = nn.AvgPool2d(2, 2)
        self.convs1 = nn.ModuleList([nn.Conv2d(1, 1, 1) for _ in range(15)])
        # Convolution layer with 45 6*6 kernels
        self.conv2 = nn.Conv2d(15, 45, 6)
        # Subsampling layer
        self.avgpool2 = nn.AvgPool2d((4, 3), (4, 3))
        self.convs2 = nn.ModuleList([nn.Conv2d(1, 1, 1) for _ in range(45)])
        # Convolution layer with 250 5*5 kernels
        self.conv3 = nn.Conv2d(45, 250, 5)
        # Fully-connected layer
        self.fc1 = nn.Linear(250, 50)

    def forward_once(self, x):
        x = self.conv1(x)
        x = self.avgpool1(x)
        # decouple channels
        xs = torch.split(x, 1, 1)
        xs = list(xs)
        for i in range(len(self.convs1)):
            xs[i] = torch.sigmoid(self.convs1[i](xs[i]))
        # concatenate tensors
        x = torch.cat(xs, 1)
        # Mimic partial connection
        x = F.dropout(x, 0.707)
        x = self.conv2(x)
        #
        x = self.avgpool2(x)
        xs = torch.split(x, 1, 1)
        xs = list(xs)
        for i in range(len(self.convs2)):
            xs[i] = torch.sigmoid(self.convs2[i](xs[i]))
        x = torch.cat(xs, 1)
        x = self.conv3(x)
        x = x.view(-1, 250 * 1 * 1)
        x = self.fc1(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
