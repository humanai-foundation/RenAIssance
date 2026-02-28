from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import models

# Define once to avoid overhead
VggOutputs = namedtuple("VggOutputs", ['fc7', 'relu5_3', 'relu4_3', 'relu3_2', 'relu2_2'])

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class CRAFT_VGG16_BN(nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super(CRAFT_VGG16_BN, self).__init__()
        
        # Modern API
        weights = models.VGG16_BN_Weights.DEFAULT if pretrained else None
        vgg_features = models.vgg16_bn(weights=weights).features

        # CORRECT INDEXING FOR VGG16_BN
        self.slice1 = nn.Sequential(*vgg_features[0:13])   # conv2_2
        self.slice2 = nn.Sequential(*vgg_features[13:23])  # conv3_3
        self.slice3 = nn.Sequential(*vgg_features[23:33])  # conv4_3
        self.slice4 = nn.Sequential(*vgg_features[33:43])  # conv5_3

        # Custom CRAFT FC layers
        self.slice5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # Initialize only new layers or all if not pretrained
        if not pretrained:
            self.apply(init_weights)
        else:
            self.slice5.apply(init_weights)

        if freeze:
            for param in self.slice1.parameters():
                param.requires_grad = False

    def forward(self, x):
        h2_2 = self.slice1(x)
        h3_3 = self.slice2(h2_2)
        h4_3 = self.slice3(h3_3)
        h5_3 = self.slice4(h4_3)
        h_fc7 = self.slice5(h5_3)
        return VggOutputs(h_fc7, h5_3, h4_3, h3_3, h2_2)
