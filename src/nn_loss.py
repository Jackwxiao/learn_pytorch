import torch
from torch.nn import L1Loss
from torch import nn

# loss 函数需要注意输入输出的shape

inputs = torch.Tensor([1, 2, 3])
targets = torch.Tensor([1, 2, 5])

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = L1Loss()
result = loss(inputs, targets)
print(result)

loss_mse = nn.MSELoss()
result_mse = loss_mse(inputs, targets)
print(result_mse)

x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)