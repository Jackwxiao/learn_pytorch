import torch
from torch.nn import L1Loss

inputs = torch.Tensor([1, 2, 3])
targets = torch.Tensor([1, 2, 5])

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = L1Loss(reduction='sum')
result = loss(inputs, targets)
print(result)
