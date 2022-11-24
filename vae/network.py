import torch
import torch.nn as nn
import numpy as np
import torch.distributions

class Encoder(nn.Module):
        def __init__(self, latent_dims):
            super(Encoder, self).__init__()
            self.conv1 = nn.Conv2d(1, 8, 3, 3)
            self.norm1 = nn.BatchNorm2d(8)
            self.conv2 = nn.Conv2d(8, 16, 3, 3)
            self.norm2 = nn.BatchNorm2d(16)
            self.conv3 = nn.Conv2d(16, 32, 3, 3)
            self.norm3 = nn.BatchNorm2d(32)
            self.conv4 = nn.Conv2d(32, 64, 3, 3)
            self.norm4 = nn.BatchNorm2d(64)

            self.fc1 = nn.Linear(64+4, 512)
            self.fc2 = nn.Linear(512, 128)
            self.fc3 = nn.Linear(128, 64)
            self.fc4 = nn.Linear(64, 8)
            self.fc5 = nn.Linear(8, latent_dims)
            self.fc6 = nn.Linear(8, latent_dims)
            self.relu = nn.ReLU()
            self.kl = 0
            
        def forward(self, x, orientation):
            x = self.relu(self.norm1(self.conv1(x)))
            x = self.relu(self.norm2(self.conv2(x)))
            x = self.relu(self.norm3(self.conv3(x)))
            x = self.relu(self.norm4(self.conv4(x)))
            x = x.view(x.shape[0], -1)
            x = torch.cat((x, orientation), 1)
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.relu(self.fc3(x))
            x = self.relu(self.fc4(x))

            mu = self.fc5(x)
            sigma = torch.exp(self.fc6(x))
            z = mu + sigma * torch.randn_like(mu)
            self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
            return z, mu , sigma

class Decoder(nn.Module):
        def unflatten(self, input_arr):
            return input_arr.view(-1,128, 3, 3, 3)

        def __init__(self, latent_dims):
            super(Decoder, self).__init__()
            self.fc1 = nn.Linear(latent_dims, 64)
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
            x = self.unflatten(x)
            x = self.relu(self.norm1(self.conv1(x)))
            x = self.relu(self.norm2(self.conv2(x)))
            x = torch.sigmoid(self.conv3(x))
            return x

class VAE(nn.Module):
    def __init__(self,latent_dims):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x, orientation):
        z, mu, sigma = self.encoder(x, orientation)
        return self.decoder(z), mu, sigma

