# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 16:22:01 2019

@author: Pulkit Dixit
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np

#Initializing hyperparameters:
batch_size = 128
learn_rate = 0.0001
scheduler_step_size = 5
scheduler_gamma = 0.5
num_epochs = 100

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0,1.0)),
    transforms.ColorJitter(
            brightness=0.1*torch.randn(1),
            contrast=0.1*torch.randn(1),
            saturation=0.1*torch.randn(1),
            hue=0.1*torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

#Creating the CNN:
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        
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
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
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

model =  discriminator()
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
train_accuracy = []
test_accuracy = []

for epochs in range(num_epochs):
    #scheduler.step()
    if(epochs==50):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learn_rate/10.0
    if(epochs==75):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learn_rate/100.0
    
    correct = 0
    total = 0
    print('Current epoch: \t\t', epochs+1, '/', num_epochs)
    #print('--------------------------------------------------')
    
    #Training:
    model.train()
    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):

        if(Y_train_batch.shape[0] < batch_size):
            continue
    
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if('step' in state and state['step']>=1024):
                    state['step'] = 1000
        
        X_train_batch = Variable(X_train_batch).cuda()
        Y_train_batch = Variable(Y_train_batch).cuda()
        _, output = model(X_train_batch)
    
        loss = criterion(output, Y_train_batch)
        prediction = output.data.max(1)[1]
        accuracy = (float(prediction.eq(Y_train_batch.data).sum())/float(batch_size))*100.0 #Computing the training accuracy
        train_accuracy.append(accuracy)
        
        #Backward propagation:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_acc = np.mean(train_accuracy)
    print('Training accuracy: \t', train_acc)
    #print('--------------------------------------------------')
    
    #Testing
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch_idx, (X_test_batch, Y_test_batch) in enumerate(testloader):

            if(Y_test_batch.shape[0] < batch_size):
                continue
        
            X_test_batch = Variable(X_test_batch).cuda()
            Y_test_batch = Variable(Y_test_batch).cuda()
            _, output = model(X_test_batch)
        
            loss = criterion(output, Y_test_batch)
            prediction = output.data.max(1)[1]
            accuracy = (float(prediction.eq(Y_test_batch.data).sum())/float(batch_size))*100.0 #Computing the training accuracy
        test_accuracy.append(accuracy)
    test_acc = np.mean(test_accuracy)
    print('Test Accuracy: \t\t', test_acc)
    print('**************************************************')
    






























