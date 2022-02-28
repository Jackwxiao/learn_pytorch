import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../dataset", train=False, transform= torchvision.transforms.ToTensor(), download= True)
dataloader = DataLoader(dataset, batch_size=64)

class Wxconv2d(nn.Module):
    def __init__(self):
        super(Wxconv2d, self).__init__()
        self.conv1 = Conv2d(in_channels=3,out_channels=6, kernel_size=3,stride=1,padding=0)

    def forward(self,x):
        x = self.conv1(x)
        return x

wxconv2d = Wxconv2d()

writer = SummaryWriter("../logs_2")

step = 0
for data in dataloader:
    imgs, targets = data
    output = wxconv2d(imgs)
    writer.add_images("input",imgs,step)
    output = torch.reshape(output, (-1,3,30,30))
    writer.add_images("output", output,step)
    step = step + 1

writer.close()