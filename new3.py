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
from util.utils import printer
from model.model_partition import construct_model
from util.utils import printer1 ,printer2,printer3,printer4,printer5 
from util.utils import add_model, scale_model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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

sock_node= socket.socket()
sock_node.connect(('localhost', 52010))

listening_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listening_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listening_sock.bind(('localhost', 54470+args.edge_id))

device_sock_all_1=[]
device_sock_all_2=[]


if args.edge_id==0:
    while len(device_sock_all_1) < 2:
        print("Waiting for incoming connections...")
        listening_sock.listen(2)
        (client_sock_1, (ip, port)) = listening_sock.accept()
        print('Got connection from ', (ip,port))
        #print(client_sock_1)
        device_sock_all_1.append(client_sock_1)


if args.edge_id==1:
    sock_node_1= socket.socket()
    sock_node_1.connect(('localhost', 54470))
    while len(device_sock_all_2) < 2:
        print("Waiting for incoming connections...")
        listening_sock.listen(2)
        (client_sock_2, (ip, port)) = listening_sock.accept()
        print('Got connection from ', (ip,port))
        #print(client_sock_2)
        device_sock_all_2.append(client_sock_2)


if args.edge_id==2:
    sock_node_2=socket.socket()
    sock_node_2.connect(('localhost', 54470))
    sock_node_3=socket.socket()
    sock_node_3.connect(('localhost', 54471))

if args.edge_id==3:
    sock_node_4=socket.socket()
    sock_node_4.connect(('localhost', 54471))


 








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
    if args.edge_id==0:
        printer1("Epoch {} Testing loss: {}".format(epoch,loss/len(dataloader)))
        printer1("{}: Accuracy {}/{} ({:.0f}%)".format(dataset_name, 
                                                    correct,
                                                    len(dataloader.dataset),
                                                    100. * correct / len(dataloader.dataset)))
    if args.edge_id==1:
        printer2("Epoch {} Testing loss: {}".format(epoch,loss/len(dataloader)))
        printer2("{}: Accuracy {}/{} ({:.0f}%)".format(dataset_name, 
                                                    correct,
                                                    len(dataloader.dataset),
                                                    100. * correct / len(dataloader.dataset)))
    if args.edge_id==2:
        printer3("Epoch {} Testing loss: {}".format(epoch,loss/len(dataloader)))
        printer3("{}: Accuracy {}/{} ({:.0f}%)".format(dataset_name, 
                                                    correct,
                                                    len(dataloader.dataset),
                                                    100. * correct / len(dataloader.dataset)))
    if args.edge_id==3:
        printer4("Epoch {} Testing loss: {}".format(epoch,loss/len(dataloader)))
        printer4("{}: Accuracy {}/{} ({:.0f}%)".format(dataset_name, 
                                                    correct,
                                                    len(dataloader.dataset),
                                                    100. * correct / len(dataloader.dataset)))


    


input=[None]*len(models)
output=[None]*len(models)

def local_train(models):
    for i in range(len(models)):
        models[i].train()
    count=0
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
    
    send_model=[]
    for i in range(0,11,2):
        send_model.append(models[i])

    msg1=["CLIENT_TO_SERVER",send_model]
    send_msg(sock_node,msg1)
    
    msg2 = recv_msg(sock_node,"SERVER_TO_CLIENT")
    rec_model=msg2[1]
    left_model=[]
    des_model=[]
    for i in range(1,11,2):
        left_model.append(models[i])
    models=[]




    if args.edge_id==0:  
        msg01=["1_to_2",left_model]
        for j in range(2):
            send_msg(device_sock_all_1[j],msg01)
        msg06=recv_msg(device_sock_all_1[0],"2_to_1")
        msg07=recv_msg(device_sock_all_1[1],"2_to_1")
        rec_model_3=msg06[1]
        rec_model_4=msg07[1]
        des_model=add_model(rec_model_3,left_model)
        des_model=add_model(des_model,rec_model_4)
        des_model=scale_model(des_model,0.33333)
    

    if args.edge_id==1:
        msg02=recv_msg(sock_node_1,"1_to_2")
        rec_model_1=msg02[1]
        msg04=["2_to_1",left_model]
        send_msg(sock_node_1,msg04)
        msg08=["2_to_3",left_model]
        send_msg(device_sock_all_2[0],msg08)
        send_msg(device_sock_all_2[1],msg08)
        msg13=recv_msg(device_sock_all_2[0],"3_to_2")
        rec_model_7=msg13[1]
        msg24=recv_msg(device_sock_all_2[1],"4_to_3")
        rec_model_12=msg24[1]
        des_model=add_model(rec_model_1,left_model)
        des_model=add_model(des_model,rec_model_7)
        des_model=add_model(des_model,rec_model_12)
        des_model=scale_model(des_model,0.25)
        


    



    if args.edge_id==2:
        msg03=recv_msg(sock_node_2,"1_to_2")
        rec_model_2=msg03[1]
        msg05=["2_to_1",left_model]
        send_msg(sock_node_2,msg05)
        msg09=recv_msg(sock_node_3,"2_to_3")
        rec_model_5=msg09[1]
        msg10=["3_to_2",left_model]
        send_msg(sock_node_3,msg10)
        des_model=add_model(rec_model_2,left_model)
        des_model=add_model(des_model,rec_model_5)
        des_model=scale_model(des_model,0.3333)


    if args.edge_id==3:
        msg21=recv_msg(sock_node_4,"2_to_3")
        rec_model_8=msg21[1]
        msg22=["4_to_3",left_model]
        send_msg(sock_node_4,msg22)
        des_model=add_model(rec_model_8,left_model)
        des_model=scale_model(des_model,0.5)

    for i in range(5):
        models.append(rec_model[i])
        models.append(des_model[i])

    models.append(rec_model[5])


        



    


for i in range(300):
    local_train(models)
    test(models, testloader, "test", i)