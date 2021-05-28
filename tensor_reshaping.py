import torch

x = torch.arange(9)

x_3x3 = x.view(3, 3)
print(x_3x3)
x_3x3 = x.reshape(3, 3)
print(x_3x3)
print(x_3x3.shape)

y = x_3x3.t()
print(y)


x = torch.arange(10)
print(x.shape)


x = x.unsqueeze(1).unsqueeze(0)
print(x.shape)
print(x.squeeze(1).shape)