import torch

x= torch.arange(12)
print(x)
print(x.shape)
print(x.reshape(3,4))
print(torch.ones((3,4)))
print(torch.zeros((3,4)))

print(torch.rand(3,4))

X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(torch.cat((X, Y), dim=0))
print(torch.cat((X, Y), dim=1))

A = X.numpy()
B = torch.tensor(A)
print(type(A), type(B))