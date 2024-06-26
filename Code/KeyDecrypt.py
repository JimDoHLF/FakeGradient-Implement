import numpy as np
import torch

def getKey32():
    keyFile = open("key.txt", "r")
    keyStr = keyFile.read()
    key = np.ndarray(shape=(4096,4096), dtype=np.float32)  # Maybe change this to double?
    for i in range (4096):
        # Convert from hex to number
        temp = int(keyStr[i * 3 : i * 3 + 3], 16)
        key[i][temp] = 1
    return torch.from_numpy(key)

def getKey64():
    keyFile = open("key.txt", "r")
    keyStr = keyFile.read()
    key = np.ndarray(shape=(4096,4096), dtype=np.float64)  # Maybe change this to double?
    for i in range (4096):
        # Convert from hex to number
        temp = int(keyStr[i * 3 : i * 3 + 3], 16)
        key[i][temp] = 1
    return torch.from_numpy(key)

def encryptKey(weight, bias):
    key32 = torch.transpose(getKey32(), dim0=0, dim1=1)
    key64 = torch.transpose(getKey64(), dim0=0, dim1=1)
    print(weight.dtype)
    print(bias.dtype)
    w = torch.matmul(weight, key32)
    b = torch.matmul(bias, key32)
    return w, b

def decryptKey(output):
    key = getKey32()
    o = torch.matmul(output, key)
    return o