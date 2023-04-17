import torch
import torch.nn as nn
from functools import partial
from einops import reduce
from collections import OrderedDict

conv4x4 = partial(nn.Conv2d, kernel_size=4, stride=2, padding=1, bias=False)
trans4x4_s4 = partial(nn.ConvTranspose2d, kernel_size=4, stride=4, bias=False)
trans4x4_s1 = partial(nn.ConvTranspose2d, kernel_size=4, stride=1, bias=False)
trans4x4_s2 = partial(nn.ConvTranspose2d, kernel_size=4, stride=2, bias=False)
trans2x2_s2 = partial(nn.ConvTranspose2d, kernel_size=2, stride=2, bias=False)

class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3, spectral_norm=False):
        super().__init__()
    
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        
        self.conv1 = conv4x4(input_channels, 128)
        
        self.conv2 = conv4x4(128, 256)
        self.bn1 = nn.BatchNorm2d(256)
        self.act1 = nn.LeakyReLU(negative_slope=0.2)
        self.conv3 = conv4x4(256, 512)
        self.bn2 = nn.BatchNorm2d(512)
        self.act2 = nn.LeakyReLU(negative_slope=0.2)
        self.conv4 = conv4x4(512, 512)
        self.bn3 = nn.BatchNorm2d(512)
        self.act3 = nn.LeakyReLU(negative_slope=0.2)
        self.conv5 = conv4x4(512, 1024)
        self.bn4 = nn.BatchNorm2d(1024)
        self.act4 = nn.LeakyReLU(negative_slope=0.2)
        self.conv6 = conv4x4(1024, 1024)
        self.bn5 = nn.BatchNorm2d(1024)
        self.act5 = nn.LeakyReLU(negative_slope=0.2)
        self.fc = nn.Linear(1024, 1)
        
        if spectral_norm:
            self.conv1 = nn.utils.parametrizations.spectral_norm(self.conv1)
            self.conv2 = nn.utils.parametrizations.spectral_norm(self.conv2)
            self.conv3 = nn.utils.parametrizations.spectral_norm(self.conv3)
            self.conv4 = nn.utils.parametrizations.spectral_norm(self.conv4)
            self.conv5 = nn.utils.parametrizations.spectral_norm(self.conv5)
            self.fc = nn.utils.parametrizations.spectral_norm(self.fc)

        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        
        
        ##########       END      ##########
        x = self.act1(self.bn1(self.conv2(self.conv1(x))))
        x = self.act2(self.bn2(self.conv3(x)))
        x = self.act3(self.bn3(self.conv4(x)))
        x = self.act4(self.bn4(self.conv5(x)))
        x = reduce(x, 'b c h w -> b c', 'mean')

        return self.fc(x)


class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super().__init__()    
        self.noise_dim = noise_dim
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        self.gen = nn.Sequential(
            trans4x4_s1(noise_dim, 1024),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            trans4x4_s4(1024, 1024),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            trans4x4_s4(1024, 1024, padding=2),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            trans4x4_s1(1024, 512),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2),
            trans4x4_s2(512, 256),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),
            trans2x2_s2(256, 128),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
            trans2x2_s2(128, 3),
            nn.BatchNorm2d(3),
            nn.Tanh()
        )
        
        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        x = x[..., None, None]
        
        ##########       END      ##########
        return self.gen(x)
    

class self_attention_Generator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        pass