import torch
import torch.nn as nn
import torch.nn.functional as F
    
class autoencoder(nn.Module):
    """CNN autoencoder which maps 1024 bit input to 256 embedding"""
    
    def __init__(self):
        super(autoencoder, self).__init__()
        
        self.encoder1 = nn.Sequential(
            nn.Conv2d(16, 32, 2),
            nn.ReLU(True),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.ReLU(True),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.ReLU(True),
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(128, 256, 3),
            nn.ReLU(True),
        )
       
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3),
            nn.ReLU(True),
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3),
            nn.ReLU(True),
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3),
            nn.ReLU(True),
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 2),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        for param in self.encoder1.parameters():
            param.requires_grad = False
        for param in self.decoder4.parameters():
            param.requires_grad = False
            
        for param in self.encoder2.parameters():
            param.requires_grad = False
        for param in self.decoder3.parameters():
            param.requires_grad = False

        for param in self.encoder3.parameters():
            param.requires_grad = False
        for param in self.decoder2.parameters():
            param.requires_grad = False
            
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)
        
        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        x = self.decoder4(x)
        
        return x
    
    def encode(self, x):
        with torch.no_grad():
            x = self.encoder1(x)
            x = self.encoder2(x)
            x = self.encoder3(x)
            x = self.encoder4(x)
        
        return x