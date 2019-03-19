import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameter of CosFace:
scaler, margin = 200, 0.035

# Hyperparameter of ArcFace:
scaler_, margin_ = 200, 0.025

class DeepID(nn.Module):
    input_size = (1, 39, 31)

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
        self.local2d = nn.Conv2dLocal(in_channels=60, out_channels=80, in_height=3, in_width=2, kernel_size=2, stride=1,
                                      padding=0)
        self.fc1 = nn.Linear(360, 160)
        self.fc2 = nn.Linear(160, 160)
        self.bn = nn.BatchNorm1d(160)
        self.metric_layer = nn.Linear(160, 2, bias=False)

    def forward(self, x):
        ### Locally connected layers implemented!
        out1 = self.conv1(x)
        out2 = F.relu(self.local2d(out1))
        out1 = out1.view(out1.shape[0], -1)
        out2 = out2.view(out2.shape[0], -1)
        out1 = self.fc1(out1)  ## Fully connected 1
        out2 = self.fc2(out2)  ## Fully connected 2
        out = self.bn(torch.add(out1, out2))
        out = F.relu(out)  # Element-wise sum with ReLU
        return out
    
    def forward_logistic_loss(self, x1, x2):
        out1, out2 = self.forward(x1), self.forward(x2)
        out = self.metric_layer((out1 - out2).abs()) 
        return out

    def forward_cosine_face(self, x1, x2, y=None, s=scaler, m=margin):
        out1, out2 = self.forward(x1), self.forward(x2)
        x = (out1 - out2).abs()
        out = self.metric_layer(x)
        out /= x.norm() * self.metric_layer.weight.norm(dim=1).detach()
        if y is not None:
            idx = [list(range(out.shape[0])), y]
            out[idx] -= m
        out *= s
        return out

    def forward_arc_face(self, x1, x2, y=None, s=scaler_, m=margin_):
        out1, out2 = self.forward(x1), self.forward(x2)
        x = (out1 - out2).abs()
        out = self.metric_layer(x)
        out /= x.norm() * self.metric_layer.weight.norm(dim=1).detach()
        if y is not None:
            idx = [list(range(out.shape[0])), y]
            out[idx] = torch.cos(torch.acos(out[idx]) + m)
        out *= s
        return out

class ChopraNet(nn.Module):
    input_size = (1, 56, 46)

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
        self.metric_layer = nn.Linear(50, 2, bias=False)

    def forward(self, x):
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
    
    def forward_logistic_loss(self, x1, x2):
        out1, out2 = self.forward(x1), self.forward(x2)
        out = self.metric_layer((out1 - out2).abs()) 
        return out

    def forward_cosine_face(self, x1, x2, y=None, s=scaler, m=margin):
        out1, out2 = self.forward(x1), self.forward(x2)
        x = (out1 - out2).abs()
        out = self.metric_layer(x)
        out /= x.norm() * self.metric_layer.weight.norm(dim=1).detach()
        if y is not None:
            idx = [list(range(out.shape[0])), y]
            out[idx] -= m
        out *= s
        return out

    def forward_arc_face(self, x1, x2, y=None, s=scaler_, m=margin_):
        out1, out2 = self.forward(x1), self.forward(x2)
        x = (out1 - out2).abs()
        out = self.metric_layer(x)
        out /= x.norm() * self.metric_layer.weight.norm(dim=1).detach()
        if y is not None:
            idx = [list(range(out.shape[0])), y]
            out[idx] = torch.cos(torch.acos(out[idx]) + m)
        out *= s
        return out



class DeepFace(nn.Module):
    input_size = (3, 152, 152)

    def __init__(self):
        super(DeepFace, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 11)
        self.pool1 = nn.MaxPool2d(3, 2, 1)
        self.conv2 = nn.Conv2d(32, 16, 9)
        self.localconv1 = nn.Conv2dLocal(in_channels=16, out_channels=16, in_height=63, in_width=63, kernel_size=9,
                                         stride=1, padding=0)
        self.localconv2 = nn.Conv2dLocal(in_channels=16, out_channels=16, in_height=55, in_width=55, kernel_size=7,
                                         stride=2, padding=0)
        self.localconv3 = nn.Conv2dLocal(in_channels=16, out_channels=16, in_height=25, in_width=25, kernel_size=5,
                                         stride=1, padding=0)
        self.fc1 = nn.Linear(16*21*21, 4096)    # input dim here?
        self.metric_layer = nn.Linear(4096, 2, bias=False)

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

    def forward_logistic_loss(self, x1, x2):
        out1, out2 = self.forward(x1), self.forward(x2)
        out = self.metric_layer((out1 - out2).abs()) 
        return out

    def forward_cosine_face(self, x1, x2, y=None, s=scaler, m=margin):
        out1, out2 = self.forward(x1), self.forward(x2)
        x = (out1 - out2).abs()
        out = self.metric_layer(x)
        out /= x.norm() * self.metric_layer.weight.norm(dim=1).detach()
        if y is not None:
            idx = [list(range(out.shape[0])), y]
            out[idx] -= m
        out *= s
        return out

    def forward_arc_face(self, x1, x2, y=None, s=scaler_, m=margin_):
        out1, out2 = self.forward(x1), self.forward(x2)
        x = (out1 - out2).abs()
        out = self.metric_layer(x)
        out /= x.norm() * self.metric_layer.weight.norm(dim=1).detach()
        if y is not None:
            idx = [list(range(out.shape[0])), y]
            out[idx] = torch.cos(torch.acos(out[idx]) + m)
        out *= s
        return out
