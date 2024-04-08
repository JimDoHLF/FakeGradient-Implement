import torch
import KeyDecrypt as kd
import numpy as np

test = np.ndarray(shape=(4095,4096), dtype=np.float32)
for i in range (4095):
    test[i][i] = i + 1

bias = np.ndarray(shape=(1,4095), dtype=np.float32)
bias = torch.from_numpy(bias)

x = np.ndarray(shape=(4096,1), dtype=np.float32)
for i in range(4096):
    x[i][0] = 1
x = torch.from_numpy(x)

test = torch.from_numpy(test)
# Test encrypt
test,bias = kd.encryptKey(test,bias)
print(test.shape)

# Test split
t1,t2,t3 = torch.split(test, 1365, 0)
tcombine = torch.cat((t1,t2,t3),1)

test = kd.decryptKey(torch.matmul(test,x)) # transpose somehow

n = 0
for i in range (4095):
    if (test[i][i] == i + 1):
        n += 1

print(n)
if (n == 4095):
    print("All data retained")