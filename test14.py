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
from util.utils import add_model, scale_model, minus_model

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
listening_sock.bind(('localhost', 52390+args.edge_id))

device_sock_all_1=[]
device_sock_all_2=[]
device_sock_all_3=[]
device_sock_all_4=[]

if args.edge_id==0:
    while len(device_sock_all_2) < 4:
        print("Waiting for incoming connections...")
        listening_sock.listen(4)
        (client_sock_1, (ip, port)) = listening_sock.accept()
        print('Got connection from ', (ip,port))
        #print(client_sock_1)
        device_sock_all_2.append(client_sock_1)


if args.edge_id==1:
    sock_node_3=socket.socket()
    sock_node_3.connect(('localhost', 52390))
    while len(device_sock_all_1) < 2:
        print("Waiting for incoming connections...")
        listening_sock.listen(2)
        (client_sock_1, (ip, port)) = listening_sock.accept()
        print('Got connection from ', (ip,port))
        #print(client_sock_1)
        device_sock_all_1.append(client_sock_1)


if args.edge_id==2:
    sock_node_4=socket.socket()
    sock_node_4.connect(('localhost', 52390))
    
    sock_node_1=socket.socket()
    sock_node_1.connect(('localhost', 52391))
    while len(device_sock_all_3) < 1:
        print("Waiting for incoming connections...")
        listening_sock.listen(1)
        (client_sock_1, (ip, port)) = listening_sock.accept()
        print('Got connection from ', (ip,port))
        #print(client_sock_1)
        device_sock_all_3.append(client_sock_1)

if args.edge_id==3:
    sock_node_5=socket.socket()
    sock_node_5.connect(('localhost', 52390))
    sock_node_2=socket.socket()
    sock_node_2.connect(('localhost', 52391))
    while len(device_sock_all_4) < 1:
        print("Waiting for incoming connections...")
        listening_sock.listen(1)
        (client_sock_1, (ip, port)) = listening_sock.accept()
        print('Got connection from ', (ip,port))
        #print(client_sock_1)
        device_sock_all_4.append(client_sock_1)


if args.edge_id==4:
    sock_node_6=socket.socket()
    sock_node_6.connect(('localhost', 52390))
    sock_node_7=socket.socket()
    sock_node_7.connect(('localhost', 52392))
    sock_node_8=socket.socket()
    sock_node_8.connect(('localhost', 52393))



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

    

def parameterserver():
    if args.edge_id==0:  
        
        msg10=recv_msg(device_sock_all_2[0],"3_to_0")
        msg11=recv_msg(device_sock_all_2[1],"3_to_0")
        msg12=recv_msg(device_sock_all_2[2],"3_to_0")
        msg13=recv_msg(device_sock_all_2[3],"3_to_0")

        rec_model_10=msg10[1]
        rec_model_11=msg11[1]
        rec_model_12=msg12[1]
        rec_model_13=msg13[1]

        des_model_1=add_model(rec_model_10,rec_model_11)
        des_model_1=add_model(des_model_1,rec_model_12)
        des_model_1=add_model(des_model_1,rec_model_12)
        des_model_1=scale_model(des_model_1,0.25)
        msg10=["0_to_3",des_model_1]
        send_msg(device_sock_all_2[0],msg10)



        msg06=recv_msg(device_sock_all_2[0],"2_to_0")
        msg07=recv_msg(device_sock_all_2[1],"2_to_0")
        msg08=recv_msg(device_sock_all_2[2],"2_to_0")
        msg09=recv_msg(device_sock_all_2[3],"2_to_0")
  
        rec_model_3=msg06[1]
        rec_model_4=msg07[1]
        rec_model_5=msg08[1]
        rec_model_6=msg09[1]

        des_model=add_model(rec_model_3,rec_model_4)
        des_model=add_model(des_model,rec_model_5)
        des_model=add_model(des_model,rec_model_6)

        des_model=scale_model(des_model,0.25)
        msg01=["0_to_2",des_model]
        for i in range(4):
            send_msg(device_sock_all_2[i],msg01)

        msg15=recv_msg(device_sock_all_2[0],"4_to_0")
        msg16=recv_msg(device_sock_all_2[1],"4_to_0")
        msg17=recv_msg(device_sock_all_2[2],"4_to_0")
        msg18=recv_msg(device_sock_all_2[3],"4_to_0")

        rec_model_15=msg15[1]
        rec_model_16=msg16[1]
        rec_model_17=msg17[1]
        rec_model_18=msg18[1]

        des_model_2=add_model(rec_model_15,rec_model_16)
        des_model_2=add_model(des_model_2,rec_model_17)
        des_model_2=add_model(des_model_2,rec_model_18)
        des_model_2=scale_model(des_model_2,0.25)
        msg11=["0_to_4",des_model_2]
        send_msg(device_sock_all_2[0],msg11)





pre_model=[]
prepre_model=[]
left_model=[]

input=[None]*len(models)
output=[None]*len(models)

def local_train():
    global models
    global optimizers
    global pre_model
    global prepre_model
 
    if args.edge_id==2: 
        msg50=["3_to_0",models]
        send_msg(sock_node_4,msg50)
    
    if args.edge_id==3: 
        msg50=["3_to_0",models]
        send_msg(sock_node_5,msg50)
        
    if args.edge_id==4: 
        msg50=["3_to_0",models]
        send_msg(sock_node_6,msg50)

    for _ in range(1):
        for i in range(len(models)):
            models[i].train()
        for images, labels in trainloader:
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
    
  
    
    if args.edge_id==1:
        msg50=["3_to_0",models]
        send_msg(sock_node_3,msg50)
        msg51=recv_msg(sock_node_3,"0_to_3")
        rec_model_50=msg51[1]
        send_model=[]
        left_model=[]
        for i in range(8,11):
            left_model.append(models[i])
        for i in range(8):
            send_model.append(models[i])
        msg1=["2_to_0",send_model]
        send_msg(sock_node_3,msg1)
        
        msg2 = recv_msg(sock_node_3,"0_to_2")
        rec_model=msg2[1]
        models=[]
        for i in range(8):
            models.append(rec_model[i])
        


    if args.edge_id==2:
        send_model=[]
        left_model=[]
        for i in range(8,11):
            left_model.append(models[i])
        for i in range(8):
            send_model.append(models[i])
        msg1=["2_to_0",send_model]
        send_msg(sock_node_4,msg1)
        
        msg2 = recv_msg(sock_node_4,"0_to_2")
        rec_model=msg2[1]
        models=[]
        for i in range(8):
            models.append(rec_model[i])

    if args.edge_id==3:
        send_model=[]
        left_model=[]
        for i in range(8,11):
            left_model.append(models[i])
        for i in range(8):
            send_model.append(models[i])
        msg1=["2_to_0",send_model]
        send_msg(sock_node_5,msg1)
        
        msg2 = recv_msg(sock_node_5,"0_to_2")
        rec_model=msg2[1]
        models=[]
        for i in range(8):
            models.append(rec_model[i])


    if args.edge_id==4:
        send_model=[]
        left_model=[]
        for i in range(8,11):
            left_model.append(models[i])
        for i in range(8):
            send_model.append(models[i])
        msg1=["2_to_0",send_model]
        send_msg(sock_node_6,msg1)
        
        msg2 = recv_msg(sock_node_6,"0_to_2")
        rec_model=msg2[1]
        models=[]
        for i in range(8):
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
        for i in range(8,11):
            models.append(des_model[i-8])
        msg60=["4_to_0",models]
        send_msg(sock_node_3,msg60)
        msg61=recv_msg(sock_node_3,"0_to_4")
        rec_model_60=msg61[1]

        
    

        
        


    
        

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
        des_model=scale_model(des_model,0.3333)
        for i in range(8,11):
            models.append(des_model[i-8])
        msg60=["4_to_0",models]
        send_msg(sock_node_4,msg60)

    if args.edge_id==3:
        msg09=recv_msg(sock_node_2,"1_to_2")
        rec_model_5=msg09[1]
        msg08=["2_to_1",left_model]
        send_msg(sock_node_2,msg08)
        msg23=["3_to_4",left_model]
        send_msg(device_sock_all_4[0],msg23)
        msg24=recv_msg(device_sock_all_4[0],"4_to_3")
        rec_model_22=msg24[1]
        des_model=add_model(rec_model_5,left_model)
        des_model=add_model(des_model,rec_model_22)
        des_model=scale_model(des_model,0.3333)
        for i in range(8,11):
            models.append(des_model[i-8])
        msg60=["4_to_0",models]
        send_msg(sock_node_5,msg60)

    if args.edge_id==4:
        msg25=recv_msg(sock_node_7,"2_to_4")
        rec_model_23=msg25[1]
        msg26=recv_msg(sock_node_8,"3_to_4")
        rec_model_24=msg26[1]
        msg27=["4_to_2",left_model]
        send_msg(sock_node_7,msg27)
        msg28=["4_to_3",left_model]
        send_msg(sock_node_8,msg28)
        des_model=add_model(rec_model_23,left_model)
        des_model=add_model(des_model,rec_model_24)
        des_model=scale_model(des_model,0.3333)
        for i in range(8,11):
            models.append(des_model[i-8])
        msg60=["4_to_0",models]
        send_msg(sock_node_6,msg60)
        

    for i in range(len(models)):
        models[i]=models[i].to(device_gpu)

    for i in range(len(optimizers)):
        if optimizers[i]!=None:
            optimizers[i] = optim.SGD(params = models[i].parameters(), lr = lr)


for i in range(100):
    if args.edge_id==0: 
        parameterserver()
    else:
        local_train()
        test(models, testloader, "test", i)