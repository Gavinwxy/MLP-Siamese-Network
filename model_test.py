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
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(20, 40, kernel_size=3),
            nn.BatchNorm2d(40),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(40, 60, kernel_size=3),
            nn.BatchNorm2d(60),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # Convolutional Layer 4: Totally unshared
        self.local2d = nn.Conv2dLocal(in_channels=60, out_channels=80, in_height=3, in_width=2, kernel_size=2, stride=1, padding=0)
        # Convolutional Layer 4: Normal
        self.conv2 = nn.Conv2d(60, 80, kernel_size=2)
        self.bn = nn.BatchNorm1d(160)
        self.fc1 = nn.Linear(360, 160)
        self.fc2 = nn.Linear(160, 160)

    def forward(self, x):
        ### Locally connected layers needed here !!!
        out1 = self.conv1(x)
        #out2 = F.relu(self.conv2(out1))
        out2 = F.relu(self.local2d(out1))
        out1 = out1.view(out1.shape[0], -1)
        out2 = out2.view(out2.shape[0], -1)
        out1 = self.fc1(out1) ## Fully connected 1
        out2 = self.fc2(out2) ## Fully connected 2
        out = self.bn(torch.add(out1,out2))
        out = F.relu(out) # Element-wise sum with ReLU
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = DeepID().to(device)

summary(model, (1, 39, 31))
