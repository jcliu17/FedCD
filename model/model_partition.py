import torch
import sys
import time
from torchvision import datasets, transforms
from torch import nn, optim
import numpy as np
import torchvision
import torch.nn.functional as F
import random
import os
import copy

class layer0(nn.Module):
    def __init__(self):
        super(layer0, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3,2,1)
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        return x

class layer1(nn.Module):
    def __init__(self):
        super(layer1,self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
    def forward(self,x):
        x = self.pool(x)
        return x

class layer2(nn.Module):
    def __init__(self):
        super(layer2, self).__init__()
        self.conv = nn.Conv2d(64, 192, kernel_size=3, padding=1)
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        return x

class layer3(nn.Module):
    def __init__(self):
        super(layer3,self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
    def forward(self,x):
        x = self.pool(x)
        return x


class layer4(nn.Module):
    def __init__(self):
        super(layer4, self).__init__()
        self.conv = nn.Conv2d(192, 384, kernel_size=3, padding=1)
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        return x

class layer5(nn.Module):
    def __init__(self):
        super(layer5, self).__init__()
        self.conv = nn.Conv2d(384, 256, kernel_size=3, padding=1)
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        return x

class layer6(nn.Module):
    def __init__(self):
        super(layer6, self).__init__()
        self.conv = nn.Conv2d(256, 256, kernel_size=3, padding=1)
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        return x

class layer7(nn.Module):
    def __init__(self):
        super(layer7,self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
    def forward(self,x):
        x = self.pool(x)
        x = x.view(-1,256*2*2)
        return x

class layer8(nn.Module):
    def __init__(self):
        super(layer8,self).__init__()
        self.fc = nn.Linear(256*2*2, 4096)
        self.drop = nn.Dropout(0.5)
    def forward(self,x):
        x = F.relu(self.fc(x), inplace=True)
        x = self.drop(x)
        return x

class layer9(nn.Module):
    def __init__(self):
        super(layer9,self).__init__()
        self.fc = nn.Linear(4096, 4096)
        self.drop = nn.Dropout()
    def forward(self,x):
        x = F.relu(self.fc(x), inplace=True)
        x = self.drop(x)
        return x

class layer10(nn.Module):
    def __init__(self):
        super(layer10,self).__init__()
        self.fc = nn.Linear(4096, 10)
    def forward(self,x):
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

def construct_model(lr):
    models=[]
    optimizers=[]
    for i in range(0,11):
        if i==0:
            model = layer0()
            optimizer = optim.SGD(params = model.parameters(), lr = lr)
            models.append(model)
            optimizers.append(optimizer)
        if i==1:
            model = layer1()
            optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==2:
            model = layer2()
            optimizer = optim.SGD(params = model.parameters(), lr = lr)
            models.append(model)
            optimizers.append(optimizer)
        if i==3:
            model = layer3()
            optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==4:
            model = layer4()
            optimizer = optim.SGD(params = model.parameters(), lr = lr)
            models.append(model)
            optimizers.append(optimizer)
        if i==5:
            model = layer5()
            optimizer = optim.SGD(params = model.parameters(), lr = lr)
            models.append(model)
            optimizers.append(optimizer)
        if i==6:
            model = layer6()
            optimizer = optim.SGD(params = model.parameters(), lr = lr)
            models.append(model)
            optimizers.append(optimizer)
        if i==7:
            model = layer7()
            optimizer = None
            models.append(model)
            optimizers.append(optimizer)
        if i==8:
            model = layer8()
            optimizer = optim.SGD(params = model.parameters(), lr = lr)
            models.append(model)
            optimizers.append(optimizer)
        if i==9:
            model = layer9()
            optimizer = optim.SGD(params = model.parameters(), lr = lr)
            models.append(model)
            optimizers.append(optimizer)
        if i==10:
            model = layer10()
            optimizer = optim.SGD(params = model.parameters(), lr = lr)
            models.append(model)
            optimizers.append(optimizer)
    return models,optimizers
