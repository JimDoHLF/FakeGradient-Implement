import torch
import KeyDecrypt as kd
import numpy as np

test = np.ndarray(shape=(4096,4096), dtype=np.float32)
for i in range (4096):
    test[i][i] = i + 1

bias = np.ndarray(shape=(1,4096))
bias = torch.from_numpy(bias)

test = torch.from_numpy(test)
test,bias = kd.encryptKey(test,bias)
test = kd.decryptKey(test)

n = 0
for i in range (4096):
    if (test[i][i] == i + 1):
        n += 1

print(n)
if (n == 4096):
    print("All data retained")