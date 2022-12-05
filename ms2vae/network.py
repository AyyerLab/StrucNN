import torch
from torch import nn
import torch.distributions

class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 5, 3)
        self.norm1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 5, 3)
        self.norm2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 5, 3)
        self.norm3 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(128+4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 8)
        self.fc4 = nn.Linear(8, latent_dims)
        self.fc5 = nn.Linear(8, latent_dims)
        self.relu = nn.ReLU()

    def forward(self, x, orientation):
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.relu(self.norm3(self.conv3(x)))
        x = x.view(x.shape[0], -1)
        x = torch.cat((x, orientation), 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        mu = self.fc4(x)
        logvar = self.fc5(x)
        z = self._sampling(mu,logvar)
        return z, mu , logvar
    @staticmethod
    def _sampling(mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std)
        return eps.mul(std).add_(mu)

class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.fc1 = nn.Linear(latent_dims, 64)
        self.fc2 = nn.Linear(64, 128*5*5*5)
        self.relu = nn.ReLU()
        self.conv1 = nn.ConvTranspose3d(128, 64, kernel_size=5, stride=3)
        self.norm1 = nn.BatchNorm3d(64)
        self.conv2 = nn.ConvTranspose3d(64, 32, kernel_size=5, stride=3)
        self.norm2 = nn.BatchNorm3d(32)
        self.conv3 = nn.ConvTranspose3d(32, 1, kernel_size=7, stride=5)

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
        return input_arr.view(-1, 128, 5, 5, 5)

class VAE(nn.Module):
    def __init__(self,latent_dims):
        super().__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, orientation):
        z, mu, logvar = self.encoder(x, orientation)
        return self.decoder(z), mu, logvar
