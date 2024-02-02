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
import time
import struct
from util.utils import send_msg, recv_msg, time_printer, time_count
import copy
from torch.autograd import Variable
from model.model_VGG import construct_VGG
from model.model_AlexNet import construct_AlexNet
from util.utils import printer
from model.model_partition import construct_model
from util.utils import printer1
from util.utils import add_model, scale_model

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

if True:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)

device_gpu = torch.device("cuda" if True else "cpu")

listening_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listening_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listening_sock.bind(('localhost', 52040))

device_num = 5
device_sock_all=[]

while len(device_sock_all) < device_num:
    listening_sock.listen(device_num)
    print("Waiting for incoming connections...")
    (client_sock, (ip, port)) = listening_sock.accept()
    print('Got connection from ', (ip,port))
    print(client_sock)
    device_sock_all.append(client_sock)
print("-------------------------------------------------------")

rec_models=[]
for i in range(300):
    for i in range(5):
        msg = recv_msg(device_sock_all[i],"CLIENT_TO_SERVER")
        rec_model=msg[1]
        rec_models.append(rec_model)
        print("ok")


    des_model=add_model(rec_models[0],rec_models[1])
    for i in range(2,5):
        des_model=add_model(des_model,rec_models[i])
    des_model=scale_model(des_model,0.2)
    """for para in des_model[0].parameters():
        print(para)"""
    
    for i in range(5):
        msg1=["SERVER_TO_CLIENT",des_model]
        send_msg(device_sock_all[i],msg1)










'''sock_dev=device_sock_all[0]
msg = recv_msg(sock_dev,"SERVER_TO_CLIENT")
rec_model=copy.deepcopy(msg[1])

for para in rec_model[0].parameters():
    print(para)'''