import torch
import torch.nn as nn
import torch.nn.functional as F


class FlappyConv(nn.Module):
    def __init__(self):
        super(FlappyConv, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1(256 ,2)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), (2, 2))
        x - F.max_pool2d(self.con2(x), (2, 2))
        x = F.max_pool2d(self.conv3(x), (2, 2))
        x = F.relu(torch.flatten(x, 1))
        x = self.fc1(x)
        return x
