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
from model.model_VGG9 import construct_VGG9
from util.utils import printer
from model.model_partition import construct_model
from util.utils import printer1 ,printer2,printer3,printer4,printer5 
from util.utils import add_model, scale_model

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device_gpu = torch.device("cuda" if True else "cpu")


if True:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)

parser = argparse.ArgumentParser(description='PyTorch MNIST SVM')
parser.add_argument('--edge_id', type=int, default=0, metavar='N',
                        help='edge server')

args = parser.parse_args()

print(args.edge_id)


listening_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listening_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listening_sock.bind(('localhost', 54460+args.edge_id))




if args.edge_id==5:
    sock_node_5=socket.socket()
    sock_node_5.connect(('localhost', 54460))

if args.edge_id==6:
    sock_node_6=socket.socket()
    sock_node_6.connect(('localhost', 54460))

if args.edge_id==7:
    sock_node_7=socket.socket()
    sock_node_7.connect(('localhost', 54460))


if args.edge_id==8:
    sock_node_8=socket.socket()
    sock_node_8.connect(('localhost', 54460))

edge_num=args.edge_id-1
if args.edge_id!=0:
    transform = transforms.Compose([  #transforms.Scale((227,227)),
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                        ])
    trainset = datasets.ImageFolder('/data/pcqu/Dataset/Dataset/cifar10/device_8_noniid_rate_07/device'+str(edge_num)+'/', transform=transform)
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



models, optimizers = construct_model(lr)

for i in range(len(models)):
    models[i]=models[i].to(device_gpu)


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
    if args.edge_id==1:
        printer1("Epoch {} Testing loss: {}".format(epoch,loss/len(dataloader)))
        printer1("{}: Accuracy {}/{} ({:.0f}%)".format(dataset_name, 
                                                    correct,
                                                    len(dataloader.dataset),
                                                    100. * correct / len(dataloader.dataset)))
    if args.edge_id==2:
        printer2("Epoch {} Testing loss: {}".format(epoch,loss/len(dataloader)))
        printer2("{}: Accuracy {}/{} ({:.0f}%)".format(dataset_name, 
                                                    correct,
                                                    len(dataloader.dataset),
                                                    100. * correct / len(dataloader.dataset)))
    if args.edge_id==3:
        printer3("Epoch {} Testing loss: {}".format(epoch,loss/len(dataloader)))
        printer3("{}: Accuracy {}/{} ({:.0f}%)".format(dataset_name, 
                                                    correct,
                                                    len(dataloader.dataset),
                                                    100. * correct / len(dataloader.dataset)))
    if args.edge_id==4:
        printer4("Epoch {} Testing loss: {}".format(epoch,loss/len(dataloader)))
        printer4("{}: Accuracy {}/{} ({:.0f}%)".format(dataset_name, 
                                                    correct,
                                                    len(dataloader.dataset),
                                                    100. * correct / len(dataloader.dataset)))
    if args.edge_id==5:
        printer5("Epoch {} Testing loss: {}".format(epoch,loss/len(dataloader)))
        printer5("{}: Accuracy {}/{} ({:.0f}%)".format(dataset_name, 
                                                    correct,
                                                    len(dataloader.dataset),
                                                    100. * correct / len(dataloader.dataset)))







input=[None]*len(models)
output=[None]*len(models)

def local_train():
    global models
    global optimizers
    for _ in range(15):
        count=0
        for i in range(len(models)):
            models[i].train()
        for images, labels in trainloader:
            count+=1
            if count%50==0:
                print(count)
            images= images.to(device_gpu)
            labels = labels.to(device_gpu)
            input[0] = images
            output[0] = models[0](input[0])
            input[1] = output[0].detach().requires_grad_()
            for i in range(1,len(models)):
                output[i] = models[i](input[i])
                if i<len(models)-1:
                    input[i+1] = output[i].detach().requires_grad_()
                else:
                    loss = criterion(output[i], labels)

            for i in range(0,len(models)):
                if optimizers[i] !=None:
                    optimizers[i].zero_grad()

            loss.backward()

            for i in range(len(models)-2, -1, -1):
                grad_in = input[i+1].grad
                output[i].backward(grad_in)

            for i in range(0,len(models)):
                if optimizers[i] !=None:
                    optimizers[i].step()



    if args.edge_id==5:
        msg19=["2_to_0",models]
        send_msg(sock_node_5,msg19)
        msg20=recv_msg(sock_node_5,"0_to_2")
        rec_model_12=msg20[1]
        models=rec_model_12

    if args.edge_id==6:
        msg21=["2_to_0",models]
        send_msg(sock_node_6,msg21)
        msg22=recv_msg(sock_node_6,"0_to_2")
        rec_model_13=msg22[1]
        models=rec_model_13

    if args.edge_id==7:
        msg23=["2_to_0",models]
        send_msg(sock_node_7,msg23)
        msg24=recv_msg(sock_node_7,"0_to_2")
        rec_model_14=msg24[1]
        models=rec_model_14
    
    if args.edge_id==8:
        msg25=["2_to_0",models]
        send_msg(sock_node_8,msg25)
        msg26=recv_msg(sock_node_8,"0_to_2")
        rec_model_15=msg26[1]
        models=rec_model_15
    
    
    for i in range(len(models)):
        models[i]=models[i].to(device_gpu)

    for i in range(len(optimizers)):
        if optimizers[i]!=None:
            optimizers[i] = optim.SGD(params = models[i].parameters(), lr = lr)

    


for i in range(200):
    local_train()
    test(models, testloader, "testps", i)
        