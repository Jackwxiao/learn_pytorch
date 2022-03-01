import torch
import torchvision.datasets
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../dataset",train=False,transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset,batch_size=64,drop_last=True)

class Wxiao(nn.Module):
    def __init__(self):
        super(Wxiao, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output

wxiao = Wxiao()

for data in dataloader:
    imgs,targets = data
    print(imgs.shape)
    print("+++++++++++++++++")
    # output = torch.reshape(imgs,(1,1,1,-1))
    # print(output.shape)
    # print("+++++++++++++++++")
    # [64,3,32,32] 扁平化为 196680
    output = torch.flatten(imgs)
    print(output.shape)
    # 196680 线性化 10
    output = wxiao(output)
    print(output.shape)
    print("+++++++++++++++++")

