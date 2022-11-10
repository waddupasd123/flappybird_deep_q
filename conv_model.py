import torch
import torch.nn as nn
import torch.nn.functional as F


class FlappyConv(nn.Module):
    def __init__(self):
        super(FlappyConv, self).__init__()

        self.episodes = []

        self.conv1 = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
        self.fc1 = nn.Linear(256 ,2)
        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.01, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2), padding=0, stride=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2), padding=1, stride=2)
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2), padding=0, stride=1)
        x = F.relu(torch.flatten(x, 1))
        x = self.fc1(x)
        return x
