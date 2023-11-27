import torch
import torch.nn as nn
# Making a very simple model to make sure it works


class SimpleModel(nn.Module):
    def __init__(self, num_of_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(num_of_channels, 32, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, num_of_channels, 2, stride=2),
            nn.ReLU()
        )


    def forward(self, x):
        # input shape (batch_size, num_channels, height, width)
        encode = self.encoder(x)
        decode = self.decoder(encode)
        
        return decode
    




