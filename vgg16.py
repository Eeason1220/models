import numpy as np
import torch
import torch.nn as nn


class VGG16(nn.Module):
    def __init__(self, nclasses = 11):
        super(VGG16, self).__init__()

        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0)
        )

        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0)
        )

        self.cnn_layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0)
        )

        self.cnn_layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0)
        )

        self.cnn_layer5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0)
        )
        self.fc_layer = nn.Sequential( 
            nn.Linear(4*4*512, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1000, 11)
        )
    def forward(self, x):
        x1 = self.cnn_layer1(x)
        x2 = self.cnn_layer2(x1)
        x3 = self.cnn_layer3(x2)
        x4 = self.cnn_layer4(x3)
        x5 = self.cnn_layer5(x4)
        out = torch.flatten(x5, 1, 3)
        xout = self.fc_layer(out)
        return xout
