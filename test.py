import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary


class DeepFace(nn.Module):
    input_size = (152, 152)

    def __init__(self):
        super(DeepFace, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 11)
        self.pool1 = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(32, 16, 9)
        self.localconv1 = nn.Conv2dLocal(in_channels=16, out_channels=16, in_height=63, in_width=63, kernel_size=9,
                                         stride=1, padding=0)
        self.localconv2 = nn.Conv2dLocal(in_channels=16, out_channels=16, in_height=55, in_width=55, kernel_size=7,
                                         stride=2, padding=0)
        self.localconv3 = nn.Conv2dLocal(in_channels=16, out_channels=16, in_height=25, in_width=25, kernel_size=5,
                                         stride=1, padding=0)
        self.fc1 = nn.Linear(16 * 21 * 21, 4096)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.localconv1(x))
        x = F.relu(self.localconv2(x))
        x = F.relu(self.localconv3(x))
        x = x.view(-1, 16 * 21 * 21)
        x = self.fc1(x)
        return x


model = DeepFace()
torchsummary.summary(model, (1, 152, 152))
