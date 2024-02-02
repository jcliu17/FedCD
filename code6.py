from ast import arg
from sqlite3 import connect
from ssl import SOCK_STREAM
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
import socket
import socketserver
import time
import struct
import argparse
from util.utils import printer2, send_msg, recv_msg, time_printer, time_count
import copy
from torch.autograd import Variable
from model.model_VGG import construct_VGG
from model.model_AlexNet import construct_AlexNet
from model.model_VGG9_cifar import construct_VGG9_cifar
from model.model_VGG9 import construct_VGG9 
from util.utils import printer
from model.model_partition import construct_model
from util.utils import printer1 ,printer2,printer3,printer4,printer5 
from util.utils import add_model, scale_model


os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device_gpu = torch.device("cuda" if True else "cpu")

if True:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)


sock_node_1=socket.socket()
sock_node_1.connect(('localhost', 54481))






transform = transforms.Compose([  #transforms.Scale((227,227)),
                                  transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                              ])
trainset = datasets.CIFAR10('/data/pcqu/Dataset/cifar10/train', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

transform = transforms.Compose([
                           transforms.Resize(32),
                           #transforms.CenterCrop(227),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
          ])
testset = datasets.ImageFolder('/data/pcqu/Dataset/cifar10/test', transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False)

lr=0.01
criterion = nn.NLLLoss()





partition=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

models, optimizers = construct_VGG(partition,lr)




def test(models, dataloader, dataset_name, epoch):
    for model in models:
        model.eval()
    correct = 0
    loss = 0
    with torch.no_grad():
        for data, target in dataloader:
            x=data.to(device_gpu)
            for i in range(0,len(models)):
                y = models[i](x)
                if i<len(models)-1:
                    x = y
                else:
                    loss += criterion(y, target)
                    pred = y.max(1, keepdim=True)[1]
                    correct += pred.eq(target.data.view_as(pred)).sum()
    printer3("Epoch {} Testing loss: {}".format(epoch,loss/len(dataloader)))
    printer3("{}: Accuracy {}/{} ({:.0f}%)".format(dataset_name, 
                                                    correct,
                                                    len(dataloader.dataset),
                                                    100. * correct / len(dataloader.dataset)))


input=[None]*len(models)
output=[None]*len(models)

def local_train():
    global models
    global optimizers
    for _ in range(1):
        for i in range(0, len(models)):
            models[i].train()
        for images, labels in trainloader:
            images= images.to(device_gpu)
            labels = labels.to(device_gpu)
            input[0] = images
            for i in range(0,len(models)):
                output[i] = models[i](input[i])
                if i<len(models)-1:
                    input[i+1] = output[i].detach().requires_grad_()
                else:
                    loss = criterion(output[i], labels)
            for optimizer in optimizers:
                if optimizer !=None:
                    optimizer.zero_grad()
            loss.backward()
            for i in range(len(models)-2, -1, -1):
                grad_in = input[i+1].grad
                output[i].backward(grad_in)
            for optimizer in optimizers:
                if optimizer !=None:
                    optimizer.step()

   
    msg03=recv_msg(sock_node_1,"A_to_B")
    rec_model=msg03[1]
    msg04=["B_to_A",models]
    send_msg(sock_node_1,msg04)
    models = copy.deepcopy(add_model(models, rec_model))
    models = copy.deepcopy(scale_model(models, 1.0/2))

    for i in range(len(optimizers)):
        if optimizers[i]!=None:
            optimizers[i] = optim.SGD(params = models[i].parameters(), lr = lr)

   

    

    

for i in range(100):
    local_train()
    test(models, testloader, "test", i)
    
    for i in range(len(models)):
        models[i]=models[i].to(device_gpu)
