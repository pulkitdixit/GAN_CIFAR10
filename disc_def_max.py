# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 18:58:06 2019

@author: Pulkit Dixit
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np

#Creating the CNN:
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        
        self.features = features
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 196, kernel_size = 3, stride = 1, padding = 1),
            nn.LayerNorm([196, 32, 32]),
            nn.LeakyReLU(0.1)) #output image = 196*32*32
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(196, 196, kernel_size = 3, stride = 2, padding = 1),
            nn.LayerNorm([196, 16, 16]),
            nn.LeakyReLU(0.1)) #output image = 196*16*16
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(196, 196, kernel_size = 3, stride = 1, padding = 1),
            nn.LayerNorm([196, 16, 16]),
            nn.LeakyReLU(0.1)) #output image = 196*16*16
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(196, 196, kernel_size = 3, stride = 2, padding = 1),
            nn.LayerNorm([196, 8, 8]),
            nn.LeakyReLU(0.1)) #output image = 196*8*8
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(196, 196, kernel_size = 3, stride = 1, padding = 1),
            nn.LayerNorm([196, 8, 8]),
            nn.LeakyReLU(0.1)) #output image = 196*8*8
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(196, 196, kernel_size = 3, stride = 1, padding = 1),
            nn.LayerNorm([196, 8, 8]),
            nn.LeakyReLU(0.1)) #output image = 196*8*8
        
        self.conv7 = nn.Sequential(
            nn.Conv2d(196, 196, kernel_size = 3, stride = 1, padding = 1),
            nn.LayerNorm([196, 8, 8]),
            nn.LeakyReLU(0.1)) #output image = 196*8*8
        
        self.conv8 = nn.Sequential(
            nn.Conv2d(196, 196, kernel_size = 3, stride = 2, padding = 1),
            nn.LayerNorm([196, 4, 4]),
            nn.LeakyReLU(0.1)) #output image = 196*4*4
        
        self.pool = nn.MaxPool2d(kernel_size = 4, stride = 4, padding = 0) #output image = 196*1*1
        
        self.fc1 = nn.Linear(196*1*1, 1)
        self.fc10 = nn.Linear(196*1*1, 10) #output 10 values
     
    #Forward Propagation:
    def forward(self, x, extract_features=0):
        extract_features = 2
        
        x = self.conv1(x)
        x = self.conv2(x)
        
        if(extract_features==2):
            h = torch.nn.functional.max_pool2d(x,4,4)
            h = h.view(-1, 196)
            return h
        
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        
        x = self.pool(x)
        
        #Reshaping the image to input it into the linear layer
        x = x.view(x.size(0), -1)
        
        fc1_out = self.fc1(x)
        fc10_out = self.fc10(x)
        
        return(fc1_out, fc10_out)
