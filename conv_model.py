import torch
import torch.nn as nn
import torch.nn.functional as F


class FlappyConv(nn.Module):
    def __init__(self):
        super(FlappyConv, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.fc4 = nn.Linear(7*7*64, 512)
        self.fc5 = nn.Linear(512, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
