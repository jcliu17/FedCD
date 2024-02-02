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
from util.utils import printer2, printer6, send_msg, recv_msg, time_printer, time_count
import copy
from torch.autograd import Variable
from model.model_VGG import construct_VGG
from model.model_AlexNet import construct_AlexNet
from model.model_VGG9_cifar import construct_VGG9_cifar
from model.model_VGG9 import construct_VGG9 
from util.utils import printer
from model.model_partition import construct_model
from util.utils import printer1 ,printer2,printer3,printer4,printer5,printer6 
from util.utils import add_model, scale_model

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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
listening_sock.bind(('localhost', 54430+args.edge_id))

device_sock_all_1=[]
device_sock_all_3=[]
device_sock_all_4=[]
device_sock_all_5=[]
device_sock_all_6=[]
device_sock_all_7=[]
device_sock_all_8=[]




if args.edge_id==1:
    while len(device_sock_all_1) < 4:
        print("Waiting for incoming connections...")
        listening_sock.listen(4)
        (client_sock_1, (ip, port)) = listening_sock.accept()
        print('Got connection from ', (ip,port))
        #print(client_sock_1)
        device_sock_all_1.append(client_sock_1)


if args.edge_id==2:
    
    sock_node_1=socket.socket()
    sock_node_1.connect(('localhost', 54431))
    while len(device_sock_all_3) < 3:
        print("Waiting for incoming connections...")
        listening_sock.listen(3)
        (client_sock_1, (ip, port)) = listening_sock.accept()
        print('Got connection from ', (ip,port))
        #print(client_sock_1)
        device_sock_all_3.append(client_sock_1)

if args.edge_id==3:
    sock_node_2=socket.socket()
    sock_node_2.connect(('localhost', 54431))
    sock_node_24=socket.socket()
    sock_node_24.connect(('localhost', 54432))
    while len(device_sock_all_4) < 2:
        print("Waiting for incoming connections...")
        listening_sock.listen(2)
        (client_sock_1, (ip, port)) = listening_sock.accept()
        print('Got connection from ', (ip,port))
        #print(client_sock_1)
        device_sock_all_4.append(client_sock_1)


if args.edge_id==4:
    sock_node_7=socket.socket()
    sock_node_7.connect(('localhost', 54432))
    sock_node_21=socket.socket()
    sock_node_21.connect(('localhost', 54431))
    while len(device_sock_all_5) < 2:
        print("Waiting for incoming connections...")
        listening_sock.listen(2)
        (client_sock_1, (ip, port)) = listening_sock.accept()
        print('Got connection from ', (ip,port))
        #print(client_sock_1)
        device_sock_all_5.append(client_sock_1)
    

if args.edge_id==5:
    sock_node_10=socket.socket()
    sock_node_10.connect(('localhost', 54433))
    sock_node_22=socket.socket()
    sock_node_22.connect(('localhost', 54431))
    sock_node_23=socket.socket()
    sock_node_23.connect(('localhost', 54434))
    while len(device_sock_all_6) < 1:
        print("Waiting for incoming connections...")
        listening_sock.listen(1)
        (client_sock_1, (ip, port)) = listening_sock.accept()
        print('Got connection from ', (ip,port))
        #print(client_sock_1)
        device_sock_all_6.append(client_sock_1)

if args.edge_id==6:
    sock_node_12=socket.socket()
    sock_node_12.connect(('localhost', 54434))
    sock_node_13=socket.socket()
    sock_node_13.connect(('localhost', 54435))
    sock_node_25=socket.socket()
    sock_node_25.connect(('localhost', 54432))
    sock_node_26=socket.socket()
    sock_node_26.connect(('localhost', 54433))

   

if args.edge_id==1:
    transform = transforms.Compose([  #transforms.Scale((227,227)),
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                        ])
    trainset = datasets.ImageFolder('/data/pcqu/Dataset/Dataset/cifar10/device_2_noniid_rate_095/device0/', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    transform = transforms.Compose([
                                    transforms.Resize(32),
                                    #transforms.CenterCrop(227),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
    testset = datasets.ImageFolder('/data/pcqu/Dataset/cifar10/test', transform = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False)


if args.edge_id==2:
    transform = transforms.Compose([  #transforms.Scale((227,227)),
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                        ])
    trainset = datasets.ImageFolder('/data/pcqu/Dataset/Dataset/cifar10/device_2_noniid_rate_095/device1/', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    transform = transforms.Compose([
                                    transforms.Resize(32),
                                    #transforms.CenterCrop(227),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
    testset = datasets.ImageFolder('/data/pcqu/Dataset/cifar10/test', transform = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False)

if args.edge_id==3:
    transform = transforms.Compose([  #transforms.Scale((227,227)),
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                        ])
    trainset = datasets.ImageFolder('/data/pcqu/Dataset/Dataset/cifar10/device_2_noniid_rate_095/device1/', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    transform = transforms.Compose([
                                    transforms.Resize(32),
                                    #transforms.CenterCrop(227),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
    testset = datasets.ImageFolder('/data/pcqu/Dataset/cifar10/test', transform = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False)

if args.edge_id==4:
    transform = transforms.Compose([  #transforms.Scale((227,227)),
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                        ])
    trainset = datasets.ImageFolder('/data/pcqu/Dataset/Dataset/cifar10/device_2_noniid_rate_095/device0/', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    transform = transforms.Compose([
                                    transforms.Resize(32),
                                    #transforms.CenterCrop(227),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
    testset = datasets.ImageFolder('/data/pcqu/Dataset/cifar10/test', transform = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False)

if args.edge_id==5:
    transform = transforms.Compose([  #transforms.Scale((227,227)),
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                        ])
    trainset = datasets.ImageFolder('/data/pcqu/Dataset/Dataset/cifar10/device_2_noniid_rate_095/device0/', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    transform = transforms.Compose([
                                    transforms.Resize(32),
                                    #transforms.CenterCrop(227),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
    testset = datasets.ImageFolder('/data/pcqu/Dataset/cifar10/test', transform = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False)


if args.edge_id==6:
    transform = transforms.Compose([  #transforms.Scale((227,227)),
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                        ])
    trainset = datasets.ImageFolder('/data/pcqu/Dataset/Dataset/cifar10/device_2_noniid_rate_095/device1/', transform=transform)
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
        '''for para in models[10].parameters():
            print(args.edge_id)
            print("final1")
            print(para)'''
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

def local_train(epoch):
    global models
    global optimizers
    '''for para in models[10].parameters():
        print(args.edge_id)
        print("prepre")
        print(para)
'''
    for _ in range(3):
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
    
    '''for para in models[10].parameters():
        print(args.edge_id)
        print("pre")
        print(para)
    '''

    


    if epoch%10<5:
        if args.edge_id==1:
            msg01=["1_to_2",models]
            for j in range(2):
                send_msg(device_sock_all_1[j],msg01)
            msg06=recv_msg(device_sock_all_1[0],"2_to_1")
            msg07=recv_msg(device_sock_all_1[1],"2_to_1")
            rec_model_3=msg06[1]
            rec_model_4=msg07[1]
            des_model=add_model(rec_model_3,models)
            des_model=add_model(des_model,rec_model_4)
            des_model=scale_model(des_model,0.33333)
            models=des_model
            

        if args.edge_id==2:
            msg02=recv_msg(sock_node_1,"1_to_2")
            rec_model_1=msg02[1]
            msg04=["2_to_1",models]
            send_msg(sock_node_1,msg04)
            msg21=["2_to_4",models]
            send_msg(device_sock_all_3[1],msg21)
            msg22=recv_msg(device_sock_all_3[1],"4_to_2")
            rec_model_21=msg22[1]
            des_model=add_model(rec_model_1,models)
            des_model=add_model(des_model,rec_model_21)
            des_model=scale_model(des_model,0.33333)
            models=des_model


        if args.edge_id==3:
            msg09=recv_msg(sock_node_2,"1_to_2")
            rec_model_5=msg09[1]
            msg08=["2_to_1",models]
            send_msg(sock_node_2,msg08)
            msg23=["3_to_5",models]
            send_msg(device_sock_all_4[0],msg23)
            msg24=recv_msg(device_sock_all_4[0],"5_to_3")
            rec_model_22=msg24[1]
            des_model=add_model(rec_model_5,models)
            des_model=add_model(des_model,rec_model_22)
            des_model=scale_model(des_model,0.3333)
            models=des_model

        if args.edge_id==4:
            msg30=recv_msg(sock_node_7,"2_to_4")
            rec_model_30=msg30[1]
            msg31=["4_to_2",models]
            send_msg(sock_node_7,msg31)
            msg32=["4_to_6",models]
            send_msg(device_sock_all_5[1],msg32)
            msg33=recv_msg(device_sock_all_5[1],"6_to_4")
            rec_model_31=msg33[1]
            des_model=add_model(rec_model_30,models)
            des_model=add_model(des_model,rec_model_31)
            des_model=scale_model(des_model,0.3333)
            models=des_model

        if args.edge_id==5:
            msg34=recv_msg(sock_node_10,"3_to_5")
            rec_model_32=msg34[1]
            msg35=["5_to_3",models]
            send_msg(sock_node_10,msg35)
            msg36=["5_to_6",models]
            send_msg(device_sock_all_6[0],msg36)
            msg37=recv_msg(device_sock_all_6[0],"6_to_5")
            rec_model_33=msg37[1]
            des_model=add_model(rec_model_32,models)
            des_model=add_model(des_model,rec_model_33)
            des_model=scale_model(des_model,0.3333)
            models=des_model


        if args.edge_id==6:
            msg38=recv_msg(sock_node_12,"4_to_6")
            rec_model_34=msg38[1]
            msg39=["6_to_4",models]
            send_msg(sock_node_12,msg39)
            msg40=recv_msg(sock_node_13,"5_to_6")
            rec_model_35=msg40[1]
            msg41=["6_to_5",models]
            send_msg(sock_node_13,msg41)
            des_model=add_model(rec_model_34,models)
            des_model=add_model(des_model,rec_model_35)
            des_model=scale_model(des_model,0.3333)
            models=des_model



    else:
        if args.edge_id==1:
            msg51=["1_to_4",models]
            send_msg(device_sock_all_1[2],msg51)
            send_msg(device_sock_all_1[3],msg51)
            msg52=recv_msg(device_sock_all_1[2],"4_to_1")
            msg53=recv_msg(device_sock_all_1[3],"4_to_1")
            rec_model_51=msg52[1]
            rec_model_52=msg53[1]
            des_model=add_model(rec_model_51,models)
            des_model=add_model(des_model,rec_model_52)
            des_model=scale_model(des_model,0.3333)
            models=des_model

        if args.edge_id==2:
            msg54=["2_to_3",models]
            send_msg(device_sock_all_3[0],msg54)
            send_msg(device_sock_all_3[2],msg54)
            msg55=recv_msg(device_sock_all_3[0],"3_to_2")
            msg56=recv_msg(device_sock_all_3[2],"3_to_2")
            rec_model_53=msg55[1]
            rec_model_54=msg56[1]
            des_model=add_model(rec_model_53,models)
            des_model=add_model(des_model,rec_model_54)
            des_model=scale_model(des_model,0.3333)
            models=des_model


        if args.edge_id==3:
            msg57=recv_msg(sock_node_24,"2_to_3")
            rec_model_55=msg57[1]
            msg58=["3_to_2",models]
            send_msg(sock_node_24,msg58)
            send_msg(device_sock_all_4[1],msg58)
            msg59=recv_msg(device_sock_all_4[1],"3_to_2")
            rec_model_56=msg59[1]
            des_model=add_model(rec_model_55,models)
            des_model=add_model(des_model,rec_model_56)
            des_model=scale_model(des_model,0.3333)
            models=des_model

        if args.edge_id==4:
            msg60=recv_msg(sock_node_21,"1_to_4")
            rec_model_57=msg60[1]
            msg61=["4_to_1",models]
            send_msg(sock_node_21,msg61)
            send_msg(device_sock_all_5[0],msg61)
            msg62=recv_msg(device_sock_all_5[0],"4_to_1")
            rec_model_58=msg62[1]
            des_model=add_model(rec_model_57,models)
            des_model=add_model(des_model,rec_model_58)
            des_model=scale_model(des_model,0.3333)
            models=des_model

        if args.edge_id==5:
            msg63=recv_msg(sock_node_22,"1_to_4")
            rec_model_59=msg63[1]
            msg64=recv_msg(sock_node_23,"4_to_1")
            rec_model_60=msg64[1]
            msg65=["4_to_1",models]
            send_msg(sock_node_22,msg65)
            send_msg(sock_node_23,msg65)
            des_model=add_model(rec_model_59,models)
            des_model=add_model(des_model,rec_model_60)
            des_model=scale_model(des_model,0.3333)
            models=des_model

        if args.edge_id==6:
            msg66=recv_msg(sock_node_25,"2_to_3")
            rec_model_61=msg66[1]
            msg67=recv_msg(sock_node_26,"3_to_2")
            rec_model_62=msg67[1]
            msg68=["3_to_2",models]
            send_msg(sock_node_25,msg68)
            send_msg(sock_node_26,msg68)
            des_model=add_model(rec_model_61,models)
            des_model=add_model(des_model,rec_model_62)
            des_model=scale_model(des_model,0.3333)
            models=des_model

    
        

    for i in range(len(models)):
        models[i]=models[i].to(device_gpu)

    '''for para in models[10].parameters():
        print(args.edge_id)
        print(para)
'''
    for i in range(len(optimizers)):
        if optimizers[i]!=None:
            optimizers[i] = optim.SGD(params = models[i].parameters(), lr = lr)


for i in range(200):
    local_train(i)
    test(models, testloader, "test", i)