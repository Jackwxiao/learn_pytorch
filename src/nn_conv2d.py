import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../dataset", train=False, transform= torchvision.transforms.ToTensor(), download= False)
dataloader = DataLoader(dataset, batch_size=64)

class Wxconv2d(nn.Module):
    def __init__(self):
        super(Wxconv2d, self).__init__()
        self.conv1 = Conv2d(in_channels=3,out_channels=6, kernel_size=3,stride=1,padding=0)

    def forward(self,x):
        x = self.conv1(x)
        return x

wxconv2d = Wxconv2d()

"""for data in dataloader:
    imgs, targets = data
    output = wxconv2d(imgs)
"""