import torch

A = torch.arange(20).reshape(5,4)
print(A)
#沿0轴（行）计算A元素的累积总和
print(A.cumsum(axis=0))
