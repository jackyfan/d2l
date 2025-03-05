import torch

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print(A)
# 沿0轴（行）计算A元素的累积总和
print(A.cumsum(axis=0))

x = torch.arange(4, dtype=torch.float32)
y = torch.ones(4, dtype=torch.float32)

# 乘积：相同位置的按元素乘积
print(x * y)
# 点积：相同位置的按元素乘积的和
print(torch.dot(x, y))
# 按元素乘法，然后进行求和来表示两个向量的点积
print(torch.sum(x * y))

# 矩阵和向量积
print(A.shape, x.shape)
print(torch.mv(A, x))

B = torch.ones(4, 3)
# 矩阵-矩阵乘法可以简单地称为矩阵乘法,A的行数必须等于B的行数
print(torch.mm(A, B))
# 两个矩阵的按元素乘法称为Hadamard积，A,B形状必须相同
A1 = torch.arange(12).reshape(4, 3)
print(A1 * B)

u = torch.tensor([[3.0, -2.0]])
print(torch.norm(u))

print(A)
# 证明一个矩阵A的转置的转置是A
print(A.T.T == A)
AT = torch.arange(12).reshape(4, 3)
BT = torch.arange(12).reshape(4, 3)
print((AT.T + BT.T) == (AT + BT).T)

C = torch.arange(24).reshape(2, 3, 4)
print(len(C))
print(torch.sum(C, axis=0))
print(torch.sum(C, axis=1))
print(torch.sum(C, axis=2))
