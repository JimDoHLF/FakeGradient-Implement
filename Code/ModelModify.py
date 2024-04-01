import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
import math
import torchvision.models as models
from PIL import Image
from HeatMapShow import ShowHeatMap
import AnalysisWeight as AW

import KeyDecrypt as kd

def ModifyModelVGGScale(net1,Scale,index):
    '''
        VGG modification
    '''
    TT=net1.classifier[0].weight
    print(TT.shape)
    TTA=net1.classifier[3].weight
    print(TTA.shape)
    TTB=net1.classifier[6].weight
    BiaB=net1.classifier[6].bias
    print(BiaB.shape)
    #exit()

    # save biase and weight
    r,c=TTB.shape
    BSave=np.zeros((1,r))
    for i in range(r):
        BSave[0][i]=BiaB[i].clone()
    BSaveB=BSave.copy()
    MaxBias=BSaveB.max()
    for i in range(r):
        Flag = 1
        if BSaveB[0][i] < 0:
            Flag = -1
        BSaveB[0][i] = (MaxBias - abs(BSaveB[0][i])) * Scale * Flag
    #exit()

    WSave=np.zeros((r,c))
    for i in range(r):
        for j in range(c):
            WSave[i][j]=TTB[i][j].clone()
    WSaveB=WSave.copy()
    WeightMax=WSaveB.max()

    for i in range(r):
        for j in range(c):
            Flag = 1
            if WSaveB[i][j] < 0:
                Flag = -1
            WSaveB[i][j] = (WeightMax - abs(WSaveB[i][j])) * Scale * Flag  # original
            #WSaveB[i][j]=(WeightMax-WSaveB[i][j])*Scale

    OutL=4095 # 3 character Hex
    net1.classifier[6]=nn.Linear(in_features=c, out_features=OutL, bias=True)
    NewWeight=net1.classifier[6].weight
    NewBias=net1.classifier[6].bias
    print(r,c)
    print(TTB.shape)
    U=TTB[0][0].clone()
    #TTB[0][0]=2
    print(TTB[0][0])
    print(U)
    print("======1--------------")
    with torch.no_grad():
        for i in range(r):
            for j in range(c):
                TMP=WSave[i][j].copy()
                NewWeight[i][j]=TMP

    print("======2--------------")
    with torch.no_grad():
        for i in range(r):
            for j in range(c):
                TMP = WSaveB[i][j].copy()
                NewWeight[i+r][j]=TMP
    print("======3--------------")
    with torch.no_grad():
        for i in range(r):
            #BSave[0][i]=BiaB[i].clone()
            TMP=BSave[0][i].copy()
            TMPB = BSaveB[0][i].copy()
            NewBias[i]=TMP
            NewBias[i+r] =TMPB
    print("======4--------------")
    with torch.no_grad():
        w,b = kd.encryptKey(NewWeight,NewBias)
    print("======5--------------")
    with torch.no_grad():
        # Split gradient
        w = w[index * 1365 : index * 1365 + 1365]
    #exit()
    return net1
