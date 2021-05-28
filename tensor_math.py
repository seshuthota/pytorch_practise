import torch

# Tensor Math

x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 8])

# Addition
z1 = torch.empty((3))
torch.add(x, y, out=z1)
z2 = torch.add(x, y)
z = x + y

# Subtraction
z = x - y

# Division
z = torch.true_divide(x, y)
# print(z)

# inplace operation
t = torch.zeros(3)
# t.add_(x)
# print(x)
t += x
# print(t)

# Exponentiation
z = x.pow(2)
z = x ** 2
# print(z)

# simple coparison
z = x > 0
z = x < 0

# Matrix Multiplication
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2)
# print(x3)

# Matrix Ecponentiation
matrix_exp = torch.rand(5, 5)
matrix_exp.pow(3)
# print(matrix_exp)

# Elementwise multiplication
z = x * y
# print(z)

# DotProduct
z = torch.dot(x, y)
# print(z)

# batch matrix multiplication
batch = 32
n = 10
m = 20
p = 30
tensor1 = torch.rand((batch, n, m,))
# print(tensor1.shape)
tensor2 = torch.rand((batch, m, p))
# print(tensor2.shape)
out_bmm = torch.bmm(tensor1, tensor2)
# print(out_bmm.shape)
# print(out_bmm)

# example broadcasting
x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))

z = x1 - x2
z = x1 ** x2
# print(z)

# useful opertations
sum_x = torch.sum(x, dim=0)

values, indices = torch.max(x, dim=0)
values, indices == torch.min(x, dim=0)
abs_x = torch.abs(x)
z = torch.argmax(x, dim=0)
z = torch.argmin(x, dim=0)
mean_x = torch.mean(x.float(), dim=0)
