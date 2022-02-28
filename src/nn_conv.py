import torch
import torch.nn.functional as F

input = torch.tensor([[1,2,0,3,1],
                       [0,1,2,3,1],
                       [1,2,1,0,0],
                       [5,2,3,1,1],
                       [2,1,0,1,1]])

kernel = torch.tensor([[1,2,1],
                       [0,1,0],
                       [2,1,0]])

input = torch.reshape(input, (1,1,5,5))
kernel = torch.reshape(kernel,(1,1,3,3))

output = F.conv2d(input, kernel, stride = 1)
print(output)
print("--------------")
output2 = F.conv2d(input, kernel, stride= 2)
print(output2)
print("--------------")
output3 = F.conv2d(input, kernel, stride = 1, padding = 1)
print(output3)
print("--------------")