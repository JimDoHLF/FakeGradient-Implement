import numpy as np
import torch

def getKey():
    keyFile = open("key.txt", "r")
    keyStr = keyFile.read()
    key = np.ndarray(shape=(4096,4096), dtype=np.float32)  # Maybe change this to double?
    for i in range (4096):
        # Convert from hex to number
        temp = int(keyStr[i * 3 : i * 3 + 3], 16)
        key[i][temp] = 1
    return torch.from_numpy(key)

def encryptKey(weight, bias):
    key = torch.transpose(getKey(), dim0=0, dim1=1)
    w = torch.matmul(weight, key)
    b = torch.matmul(weight, key)
    return w, b

def decryptKey(output):
    key = getKey()
    o = torch.matmul(output, key)
    return o