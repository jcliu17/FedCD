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
from util.utils import printer1,printer2,printer3,printer4,printer5,printer6,printer7, printer8, printer9, printer10 
from util.utils import add_model, scale_model

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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
listening_sock.bind(('localhost', 54350+args.edge_id))

device_sock_all_1=[]
device_sock_all_2=[]
device_sock_all_3=[]
device_sock_all_4=[]
device_sock_all_5=[]
device_sock_all_6=[]
device_sock_all_7=[]
device_sock_all_8=[]
device_sock_all_9=[]
device_sock_all_10=[]

if args.edge_id==0:
    while len(device_sock_all_2) < 10:
        print("Waiting for incoming connections...")
        listening_sock.listen(10)
        (client_sock_1, (ip, port)) = listening_sock.accept()
        print('Got connection from ', (ip,port))
        #print(client_sock_1)
        device_sock_all_2.append(client_sock_1)


if args.edge_id==1:
    sock_node_3=socket.socket()
    sock_node_3.connect(('localhost', 54350))
    while len(device_sock_all_1) < 2:
        print("Waiting for incoming connections...")
        listening_sock.listen(2)
        (client_sock_1, (ip, port)) = listening_sock.accept()
        print('Got connection from ', (ip,port))
        #print(client_sock_1)
        device_sock_all_1.append(client_sock_1)


if args.edge_id==2:
    sock_node_4=socket.socket()
    sock_node_4.connect(('localhost', 54350))
    
    sock_node_1=socket.socket()
    sock_node_1.connect(('localhost', 54351))
    while len(device_sock_all_3) < 1:
        print("Waiting for incoming connections...")
        listening_sock.listen(1)
        (client_sock_1, (ip, port)) = listening_sock.accept()
        print('Got connection from ', (ip,port))
        #print(client_sock_1)
        device_sock_all_3.append(client_sock_1)

if args.edge_id==3:
    sock_node_5=socket.socket()
    sock_node_5.connect(('localhost', 54350))
    sock_node_2=socket.socket()
    sock_node_2.connect(('localhost', 54351))
    while len(device_sock_all_4) < 1:
        print("Waiting for incoming connections...")
        listening_sock.listen(1)
        (client_sock_1, (ip, port)) = listening_sock.accept()
        print('Got connection from ', (ip,port))
        #print(client_sock_1)
        device_sock_all_4.append(client_sock_1)


if args.edge_id==4:
    sock_node_6=socket.socket()
    sock_node_6.connect(('localhost', 54350))
    sock_node_7=socket.socket()
    sock_node_7.connect(('localhost', 54352))
    while len(device_sock_all_5) < 1:
        print("Waiting for incoming connections...")
        listening_sock.listen(1)
        (client_sock_1, (ip, port)) = listening_sock.accept()
        print('Got connection from ', (ip,port))
        #print(client_sock_1)
        device_sock_all_5.append(client_sock_1)
    

if args.edge_id==5:
    sock_node_9=socket.socket()
    sock_node_9.connect(('localhost', 54350))
    sock_node_10=socket.socket()
    sock_node_10.connect(('localhost', 54353))
    while len(device_sock_all_6) < 1:
        print("Waiting for incoming connections...")
        listening_sock.listen(1)
        (client_sock_1, (ip, port)) = listening_sock.accept()
        print('Got connection from ', (ip,port))
        #print(client_sock_1)
        device_sock_all_6.append(client_sock_1)

if args.edge_id==6:
    sock_node_11=socket.socket()
    sock_node_11.connect(('localhost', 54350))
    sock_node_12=socket.socket()
    sock_node_12.connect(('localhost', 54354))
    while len(device_sock_all_7) < 1:
        print("Waiting for incoming connections...")
        listening_sock.listen(1)
        (client_sock_1, (ip, port)) = listening_sock.accept()
        print('Got connection from ', (ip,port))
        #print(client_sock_1)
        device_sock_all_7.append(client_sock_1)


if args.edge_id==7:
    sock_node_13=socket.socket()
    sock_node_13.connect(('localhost', 54350))
    sock_node_14=socket.socket()
    sock_node_14.connect(('localhost', 54355))
    while len(device_sock_all_8) < 1:
        print("Waiting for incoming connections...")
        listening_sock.listen(1)
        (client_sock_1, (ip, port)) = listening_sock.accept()
        print('Got connection from ', (ip,port))
        #print(client_sock_1)
        device_sock_all_8.append(client_sock_1)


if args.edge_id==8:
    sock_node_15=socket.socket()
    sock_node_15.connect(('localhost', 54350))
    sock_node_16=socket.socket()
    sock_node_16.connect(('localhost', 54356))
    while len(device_sock_all_9) < 1:
        print("Waiting for incoming connections...")
        listening_sock.listen(1)
        (client_sock_1, (ip, port)) = listening_sock.accept()
        print('Got connection from ', (ip,port))
        #print(client_sock_1)
        device_sock_all_9.append(client_sock_1)

if args.edge_id==9:
    sock_node_18=socket.socket()
    sock_node_18.connect(('localhost', 54350))
    sock_node_19=socket.socket()
    sock_node_19.connect(('localhost', 54357))
    while len(device_sock_all_10) < 1:
        print("Waiting for incoming connections...")
        listening_sock.listen(1)
        (client_sock_1, (ip, port)) = listening_sock.accept()
        print('Got connection from ', (ip,port))
        #print(client_sock_1)
        device_sock_all_10.append(client_sock_1)

if args.edge_id==10:
    sock_node_20=socket.socket()
    sock_node_20.connect(('localhost', 54350))
    sock_node_21=socket.socket()
    sock_node_21.connect(('localhost', 54358))
    sock_node_22=socket.socket()
    sock_node_22.connect(('localhost', 54359))



   

edge_num=args.edge_id-1
if args.edge_id!=0 and args.edge_id<=8:
    transform = transforms.Compose([  #transforms.Scale((227,227)),
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                    ])
    trainset = datasets.ImageFolder('/data/pcqu/Dataset/Dataset/cifar10/device_8_noniid_rate_05/device'+str(edge_num)+'/', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    transform = transforms.Compose([
                                transforms.Resize(32),
                                #transforms.CenterCrop(227),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
    testset = datasets.ImageFolder('/data/pcqu/Dataset/cifar10/test', transform = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False)
if args.edge_id==9:
    transform = transforms.Compose([  #transforms.Scale((227,227)),
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                    ])
    trainset = datasets.ImageFolder('/data/pcqu/Dataset/Dataset/cifar10/device_8_noniid_rate_05/device6/', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    transform = transforms.Compose([
                                transforms.Resize(32),
                                #transforms.CenterCrop(227),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
    testset = datasets.ImageFolder('/data/pcqu/Dataset/cifar10/test', transform = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False)

if args.edge_id==10:
    transform = transforms.Compose([  #transforms.Scale((227,227)),
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                    ])
    trainset = datasets.ImageFolder('/data/pcqu/Dataset/Dataset/cifar10/device_8_noniid_rate_05/device7/', transform=transform)
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
    if args.edge_id==6:
        '''for para in models[10].parameters():
            print(args.edge_id)
            print("final1")
            print(para)'''
        printer6("Epoch {} Testing loss: {}".format(epoch,loss/len(dataloader)))
        printer6("{}: Accuracy {}/{} ({:.0f}%)".format(dataset_name, 
                                                    correct,
                                                    len(dataloader.dataset),
                                                    100. * correct / len(dataloader.dataset)))
    if args.edge_id==7:
        printer7("Epoch {} Testing loss: {}".format(epoch,loss/len(dataloader)))
        printer7("{}: Accuracy {}/{} ({:.0f}%)".format(dataset_name, 
                                                    correct,
                                                    len(dataloader.dataset),
                                                    100. * correct / len(dataloader.dataset)))
    if args.edge_id==8:
        printer8("Epoch {} Testing loss: {}".format(epoch,loss/len(dataloader)))
        printer8("{}: Accuracy {}/{} ({:.0f}%)".format(dataset_name, 
                                                    correct,
                                                    len(dataloader.dataset),
                                                    100. * correct / len(dataloader.dataset)))
    if args.edge_id==9:
        printer9("Epoch {} Testing loss: {}".format(epoch,loss/len(dataloader)))
        printer9("{}: Accuracy {}/{} ({:.0f}%)".format(dataset_name, 
                                                    correct,
                                                    len(dataloader.dataset),
                                                    100. * correct / len(dataloader.dataset)))

    if args.edge_id==10:
        printer10("Epoch {} Testing loss: {}".format(epoch,loss/len(dataloader)))
        printer10("{}: Accuracy {}/{} ({:.0f}%)".format(dataset_name, 
                                                    correct,
                                                    len(dataloader.dataset),
                                                    100. * correct / len(dataloader.dataset)))

def parameterserver():
    if args.edge_id==0:  
        msg06=recv_msg(device_sock_all_2[0],"2_to_0")
        msg07=recv_msg(device_sock_all_2[1],"2_to_0")
        msg08=recv_msg(device_sock_all_2[2],"2_to_0")
        msg09=recv_msg(device_sock_all_2[3],"2_to_0")
        msg10=recv_msg(device_sock_all_2[4],"2_to_0")
        msg11=recv_msg(device_sock_all_2[5],"2_to_0")
        msg12=recv_msg(device_sock_all_2[6],"2_to_0")
        msg13=recv_msg(device_sock_all_2[7],"2_to_0")
        msg14=recv_msg(device_sock_all_2[8],"2_to_0")
        msg15=recv_msg(device_sock_all_2[9],"2_to_0")
  
        rec_model_3=msg06[1]
        rec_model_4=msg07[1]
        rec_model_5=msg08[1]
        rec_model_6=msg09[1]
        rec_model_7=msg10[1]
        rec_model_8=msg11[1]
        rec_model_9=msg12[1]
        rec_model_10=msg13[1]
        rec_model_11=msg14[1]
        rec_model_12=msg15[1]
    
        des_model=add_model(rec_model_3,rec_model_4)
        des_model=add_model(des_model,rec_model_5)
        des_model=add_model(des_model,rec_model_6)
        des_model=add_model(des_model,rec_model_7)
        des_model=add_model(des_model,rec_model_8)
        des_model=add_model(des_model,rec_model_9)
        des_model=add_model(des_model,rec_model_10)
        des_model=add_model(des_model,rec_model_11)
        des_model=add_model(des_model,rec_model_12)

        des_model=scale_model(des_model,1.0/10)
        msg01=["0_to_2",des_model]
        for i in range(10):
            send_msg(device_sock_all_2[i],msg01)



input=[None]*len(models)
output=[None]*len(models)

def local_train():
    global models
    global optimizers
    '''for para in models[10].parameters():
        print(args.edge_id)
        print("prepre")
        print(para)
'''
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
    
    '''for para in models[10].parameters():
        print(args.edge_id)
        print("pre")
        print(para)
    '''
    if args.edge_id==1:
        send_model=[]
        for i in range(9):
            send_model.append(models[i])

        msg1=["2_to_0",send_model]
        send_msg(sock_node_3,msg1)
        
        msg2 = recv_msg(sock_node_3,"0_to_2")
        rec_model=msg2[1]
        left_model=[]
        for i in range(9,11):
            left_model.append(models[i])
        models=[]
        for i in range(9):
            models.append(rec_model[i])

    if args.edge_id==2:
        send_model=[]
        for i in range(9):
            send_model.append(models[i])

        msg1=["2_to_0",send_model]
        send_msg(sock_node_4,msg1)
        
        msg2 = recv_msg(sock_node_4,"0_to_2")
        rec_model=msg2[1]
        left_model=[]
        for i in range(9,11):
            left_model.append(models[i])
        models=[]
        for i in range(9):
            models.append(rec_model[i])

    if args.edge_id==3:
        send_model=[]
        for i in range(9):
            send_model.append(models[i])

        msg1=["2_to_0",send_model]
        send_msg(sock_node_5,msg1)
        
        msg2 = recv_msg(sock_node_5,"0_to_2")
        rec_model=msg2[1]
        left_model=[]
        for i in range(9,11):
            left_model.append(models[i])
        models=[]
        for i in range(9):
            models.append(rec_model[i])


    if args.edge_id==4:
        send_model=[]
        for i in range(9):
            send_model.append(models[i])

        msg1=["2_to_0",send_model]
        send_msg(sock_node_6,msg1)
        
        msg2 = recv_msg(sock_node_6,"0_to_2")
        rec_model=msg2[1]
        left_model=[]
        for i in range(9,11):
            left_model.append(models[i])
        models=[]
        for i in range(9):
            models.append(rec_model[i])

    if args.edge_id==5:
        send_model=[]
        for i in range(9):
            send_model.append(models[i])

        msg1=["2_to_0",send_model]
        send_msg(sock_node_9,msg1)
        
        msg2 = recv_msg(sock_node_9,"0_to_2")
        rec_model=msg2[1]
        left_model=[]
        for i in range(9,11):
            left_model.append(models[i])
        models=[]
        for i in range(9):
            models.append(rec_model[i])
    
    if args.edge_id==6:
        send_model=[]
        for i in range(9):
            send_model.append(models[i])

        msg1=["2_to_0",send_model]
        send_msg(sock_node_11,msg1)
        
        msg2 = recv_msg(sock_node_11,"0_to_2")
        rec_model=msg2[1]
        left_model=[]
        for i in range(9,11):
            left_model.append(models[i])
        models=[]
        for i in range(9):
            models.append(rec_model[i])

    if args.edge_id==7:
        send_model=[]
        for i in range(9):
            send_model.append(models[i])

        msg1=["2_to_0",send_model]
        send_msg(sock_node_13,msg1)
        
        msg2 = recv_msg(sock_node_13,"0_to_2")
        rec_model=msg2[1]
        left_model=[]
        for i in range(9,11):
            left_model.append(models[i])
        models=[]
        for i in range(9):
            models.append(rec_model[i])

    if args.edge_id==8:
        send_model=[]
        for i in range(9):
            send_model.append(models[i])

        msg1=["2_to_0",send_model]
        send_msg(sock_node_15,msg1)
        
        msg2 = recv_msg(sock_node_15,"0_to_2")
        rec_model=msg2[1]
        left_model=[]
        for i in range(9,11):
            left_model.append(models[i])
        models=[]
        for i in range(9):
            models.append(rec_model[i])

    if args.edge_id==9:
        send_model=[]
        for i in range(9):
            send_model.append(models[i])

        msg1=["2_to_0",send_model]
        send_msg(sock_node_18,msg1)
        
        msg2 = recv_msg(sock_node_18,"0_to_2")
        rec_model=msg2[1]
        left_model=[]
        for i in range(9,11):
            left_model.append(models[i])
        models=[]
        for i in range(9):
            models.append(rec_model[i])

    if args.edge_id==10:
        send_model=[]
        for i in range(9):
            send_model.append(models[i])

        msg1=["2_to_0",send_model]
        send_msg(sock_node_20,msg1)
        
        msg2 = recv_msg(sock_node_20,"0_to_2")
        rec_model=msg2[1]
        left_model=[]
        for i in range(9,11):
            left_model.append(models[i])
        models=[]
        for i in range(9):
            models.append(rec_model[i])
    


    if args.edge_id==1:
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
        for i in range(9,11):
            models.append(des_model[i-9])
        

    if args.edge_id==2:
        msg02=recv_msg(sock_node_1,"1_to_2")
        rec_model_1=msg02[1]
        msg04=["2_to_1",left_model]
        send_msg(sock_node_1,msg04)
        msg21=["2_to_4",left_model]
        send_msg(device_sock_all_3[0],msg21)
        msg22=recv_msg(device_sock_all_3[0],"4_to_2")
        rec_model_21=msg22[1]
        des_model=add_model(rec_model_1,left_model)
        des_model=add_model(des_model,rec_model_21)
        des_model=scale_model(des_model,0.33333)
        for i in range(9,11):
            models.append(des_model[i-9])


    if args.edge_id==3:
        msg09=recv_msg(sock_node_2,"1_to_2")
        rec_model_5=msg09[1]
        msg08=["2_to_1",left_model]
        send_msg(sock_node_2,msg08)
        msg23=["3_to_5",left_model]
        send_msg(device_sock_all_4[0],msg23)
        msg24=recv_msg(device_sock_all_4[0],"5_to_3")
        rec_model_22=msg24[1]
        des_model=add_model(rec_model_5,left_model)
        des_model=add_model(des_model,rec_model_22)
        des_model=scale_model(des_model,0.3333)
        for i in range(9,11):
            models.append(des_model[i-9])

    if args.edge_id==4:
        msg30=recv_msg(sock_node_7,"2_to_4")
        rec_model_30=msg30[1]
        msg31=["4_to_2",left_model]
        send_msg(sock_node_7,msg31)
        msg32=["4_to_6",left_model]
        send_msg(device_sock_all_5[0],msg32)
        msg33=recv_msg(device_sock_all_5[0],"6_to_4")
        rec_model_31=msg33[1]
        des_model=add_model(rec_model_30,left_model)
        des_model=add_model(des_model,rec_model_31)
        des_model=scale_model(des_model,0.3333)
        for i in range(9,11):
            models.append(des_model[i-9])

    if args.edge_id==5:
        msg34=recv_msg(sock_node_10,"3_to_5")
        rec_model_32=msg34[1]
        msg35=["5_to_3",left_model]
        send_msg(sock_node_10,msg35)
        msg36=["5_to_7",left_model]
        send_msg(device_sock_all_6[0],msg36)
        msg37=recv_msg(device_sock_all_6[0],"7_to_5")
        rec_model_33=msg37[1]
        des_model=add_model(rec_model_32,left_model)
        des_model=add_model(des_model,rec_model_33)
        des_model=scale_model(des_model,0.3333)
        for i in range(9,11):
            models.append(des_model[i-9])


    if args.edge_id==6:
        msg38=recv_msg(sock_node_12,"4_to_6")
        rec_model_34=msg38[1]
        msg39=["6_to_4",left_model]
        send_msg(sock_node_12,msg39)
        msg40=["6_to_8",left_model]
        send_msg(device_sock_all_7[0],msg40)
        msg41=recv_msg(device_sock_all_7[0],"8_to_6")
        rec_model_35=msg41[1]
        des_model=add_model(rec_model_34,left_model)
        des_model=add_model(des_model,rec_model_35)
        des_model=scale_model(des_model,0.3333)
        for i in range(9,11):
            models.append(des_model[i-9])

    if args.edge_id==7:
        msg42=recv_msg(sock_node_14,"5_to_7")
        rec_model_36=msg42[1]
        msg43=["7_to_5",left_model]
        send_msg(sock_node_14,msg43)
        msg44=["7_to_9",left_model]
        send_msg(device_sock_all_8[0],msg44)
        msg45=recv_msg(device_sock_all_8[0],"9_to_7")
        rec_model_37=msg45[1]
        des_model=add_model(rec_model_37,left_model)
        des_model=add_model(des_model,rec_model_36)
        des_model=scale_model(des_model,0.3333)
        for i in range(9,11):
            models.append(des_model[i-9])

    if args.edge_id==8:
        msg46=recv_msg(sock_node_16,"6_to_8")
        rec_model_38=msg46[1]
        msg48=["8_to_6",left_model]
        send_msg(sock_node_16,msg48)
        msg49=["8_to_10",left_model]
        send_msg(device_sock_all_9[0],msg49)
        msg50=recv_msg(device_sock_all_9[0],"10_to_8")
        rec_model_39=msg50[1]
        des_model=add_model(rec_model_39,left_model)
        des_model=add_model(des_model,rec_model_38)
        des_model=scale_model(des_model,0.3333)
        for i in range(9,11):
            models.append(des_model[i-9])


    if args.edge_id==9:
        msg51=recv_msg(sock_node_19,"7_to_9")
        rec_model_40=msg51[1]
        msg52=["9_to_7",left_model]
        send_msg(sock_node_19,msg52)
        msg53=["9_to_10",left_model]
        send_msg(device_sock_all_10[0],msg53)
        msg54=recv_msg(device_sock_all_10[0],"10_to_9")
        rec_model_41=msg54[1]
        des_model=add_model(rec_model_40,left_model)
        des_model=add_model(des_model,rec_model_41)
        des_model=scale_model(des_model,0.3333)
        for i in range(9,11):
            models.append(des_model[i-9])
    
    if args.edge_id==10:
        msg54=recv_msg(sock_node_21,"8_to_10")
        rec_model_42=msg54[1]
        msg55=recv_msg(sock_node_22,"9_to_10")
        rec_model_43=msg55[1]
        msg56=["10_to_8",left_model]
        send_msg(sock_node_21,msg56)
        msg57=["10_to_9",left_model]
        send_msg(sock_node_22,msg57)
        des_model=add_model(rec_model_42,left_model)
        des_model=add_model(des_model,rec_model_43)
        des_model=scale_model(des_model,0.3333)
        for i in range(9,11):
            models.append(des_model[i-9])
    
    
        

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
    if args.edge_id==0: 
        parameterserver()
    else:
        local_train()
        test(models, testloader, "testnoniidcdnormal", i)