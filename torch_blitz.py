from __future__ import print_function
import torch

#construct a 5x3 matrix, uninitialized
x = torch.Tensor(5, 3)
print(x)

#construct a 5x3 matrix, randomly initialized
x = torch.rand(5, 3)
print(x)

#get the size of tensor
print(x.size())

#operations
#addition
y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))

res = torch.Tensor(5, 3)
torch.add(x, y, out=res)
print(res)

#in-place addition
y.add_(x)
print(y)

#convert tensor to numpy
a = torch.ones(5)
print(a)
b = a.numpy()
print(a, b)
a.add_(1)
print(a, b)

#convert numpy array to torch tensor
import numpy as np 
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a, b)

#cuda tensor using .cuda function
if torch.cuda.is_available():
	x = x.cuda()
	y = y.cuda()
	x + y
	print("cuda works")