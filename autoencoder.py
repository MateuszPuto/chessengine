import torch
import torch.nn as nn
import torch.nn.functional as F

class autoencoder(nn.Module):
    """CNN autoencoder which maps 1024 bit input to 256 embedding"""
    
    def __init__(self):
        super(autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(16, 32, 2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(32, 64, 2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(64, 128, 2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(128, 256, 2),
            nn.ReLU(True),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 2),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x
    
    def encode(self, x):
        return self.encoder(x)