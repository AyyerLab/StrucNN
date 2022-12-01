'''Module defining network architectures'''
import torch
from torch import nn

class DecoderCNNSim(nn.Module):
    '''Convolution NN to predict sliced intensities for given particle size'''
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 128*3*3*3)
        self.relu = nn.ReLU()
        self.conv1 = nn.ConvTranspose3d(128, 64, kernel_size=3, stride=3)
        self.norm1 = nn.BatchNorm3d(64)
        self.conv2 = nn.ConvTranspose3d(64, 32, kernel_size=3, stride=3)
        self.norm2 = nn.BatchNorm3d(32)
        self.conv3 = nn.ConvTranspose3d(32, 1, kernel_size=9, stride=6)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self._unflatten(x)
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        x = torch.sigmoid(self.conv3(x))
        return x

    @staticmethod
    def _unflatten(input_arr):
        return input_arr.view(-1, 128, 3, 3, 3)
