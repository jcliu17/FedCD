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


for i in range(30):
    if i<10:
        print("ok")
    if i>=10 and i<20:
        print(i)