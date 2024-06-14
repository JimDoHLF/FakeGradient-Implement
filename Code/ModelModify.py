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

def ModifyModelVGGScale(net,net1,net2,net3,Scale):
    '''
        VGG modification
    '''
    TT=net.classifier[0].weight
    print(TT.shape)
    TTA=net.classifier[3].weight
    print(TTA.shape)
    TTB=net.classifier[6].weight
    print(TTB.shape)
    BiaB=net.classifier[6].bias
    print(BiaB.shape)
    #exit()

    # save bias and weight
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

    net.classifier[6] = nn.Linear(in_features=c, out_features=OutL, bias=True)
    wb = net.classifier[6].weight
    bb = net.classifier[6].bias

    net1.classifier[6]=nn.Linear(in_features=c, out_features=int(OutL/3), bias=True)
    w1=net1.classifier[6].weight
    b1=net1.classifier[6].bias
    net2.classifier[6]=nn.Linear(in_features=c, out_features=int(OutL/3), bias=True)
    w2=net2.classifier[6].weight
    b2=net2.classifier[6].bias
    net3.classifier[6]=nn.Linear(in_features=c, out_features=int(OutL/3), bias=True)
    w3=net3.classifier[6].weight
    b3=net3.classifier[6].bias

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
                wb[i][j]=TMP
        """
        for i in range(OutL - r):
            for j in range(c):
                TMP = WSaveB[i][j].copy()
                wb[i+r][j]=TMP
        """
    print("======3--------------")
    with torch.no_grad():
        for i in range(r):
            #BSave[0][i]=BiaB[i].clone()
            TMP=BSave[0][i].copy()
            bb[i]=TMP
        """
        for i in range(OutL - r):
            TMPB = BSaveB[0][i].copy()
            bb[i+r] =TMPB
        """

    print("======4--------------")
    with torch.no_grad():
        wb,bb = kd.encryptKey(wb,bb)
        # Split gradient
        w1, w2, w3 = torch.split(wb, 1365, 0)
        b1, b2, b3 = torch.split(bb, 1365, 0)
        net1.classifier[6].weight = nn.Parameter(w1,False)
        net2.classifier[6].weight = nn.Parameter(w2,False)
        net3.classifier[6].weight = nn.Parameter(w3,False)
        net1.classifier[6].bias = nn.Parameter(b1,False)
        net2.classifier[6].bias = nn.Parameter(b2,False)
        net3.classifier[6].bias = nn.Parameter(b3,False)
    print("======5--------------")
    #exit()

    # Verify integrity
    wMatch = 0
    for i in range(1000):
        for n in range(4096):
            if (net1.classifier[6].weight[i][n] == wb[i][n]):
                wMatch += 1
    print("Total weight match:")
    print(wMatch)
    print("Required weight match:")
    print(4096 * 1000)
    print("Weight Verification:")
    print(wMatch == (4096 * 1000))
    bMatch = 0
    for i in range(1000):
        if (net1.classifier[6].bias[i] == bb[i]):
            bMatch += 1
    print("Total bias match:")
    print(bMatch)
    print("Required bias match:")
    print(1000)
    print("Weight Verification:")
    print(bMatch == 1000)
    print("---------------")

    return net, net1, net2, net3
