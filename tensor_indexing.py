import torch

batch_size = 10
features = 25
x = torch.rand((batch_size, features))
print(x[0])
print(x[:, 0])

print(x[2, 0:10])

# Fancy indexing
x = torch.arange(10)
indices = [2, 5, 6]
print(x)
print(x[indices])

x = torch.rand((3, 5))
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])

# more advanced
x = torch.arange(10)
print(x[(x < 2) | (x > 8)])

#Operations
print(torch.where(x >  5, x, x *  2))
print(x.ndimension())
print(x.numel())

