import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

# loss ： 1.计算实际输出和目标之间的差距
#         2. 为更新输出提出一定的依据(反向传播), grad

dataset = torchvision.datasets.CIFAR10("../dataset",train=False,transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=1)

# 自定义神经网络
class Wxiao(nn.Module):
    def __init__(self):
        super(Wxiao, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

loss = nn.CrossEntropyLoss()# 交叉熵损失函数
wxiao = Wxiao()
for data in dataloader:
    imgs, targets = data
    outputs = wxiao(imgs)
    result_loss = loss(outputs, targets)# 神经网络输出和真实值的误差