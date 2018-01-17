#https://zhuanlan.zhihu.com/p/26871672
#install pytorch in window 10

#cuda test
import torch
x = torch.Tensor([1.0])
xx = x.cuda()
print(xx)

#cudnn test
from torch.backends import cudnn
print(cudnn.is_acceptable(xx))