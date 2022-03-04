import torch
from torch import nn
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

# 准备数据集
train_data = torchvision.datasets.CIFAR10("../dataset", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为： {}".format(train_data_size))
print("测试数据集的长度为： {}".format(test_data_size))

# 加载数据
train_data_loader = DataLoader(train_data, batch_size=64)
test_data_loader = DataLoader(test_data, batch_size=64)

# 定义网络模型
class Wxiao(nn.Module):
    def __init__(self):
        super(Wxiao, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x
# 创建网络
wxiao = Wxiao()
# 使用gpu训练
wxiao = wxiao.cuda()

# 损失函数
loss_fun = nn.CrossEntropyLoss()
loss_fun = loss_fun.cuda()

# 优化器
learning_rate = 1e-2
optimzer = torch.optim.SGD(wxiao.parameters(), lr=learning_rate)

# 设置训练网络的参数
# 次数
total_train_step = 0 # 训练次数
total_test_step = 0 # 测试次数
# 训练轮数
epoch = 10

#  添加tensorboard
writer = SummaryWriter("../logs_train")
start_time = time.time()
for i in range(epoch):
    print("第{}轮训练开始".format(i+1))
    # 训练
    wxiao.train()
    for data in train_data_loader:
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = wxiao(imgs)
        loss = loss_fun(outputs, targets)

        # 优化调优
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("训练次数: {}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

# 测试
    wxiao.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad(): #测试不需要梯度
        for data in test_data_loader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = wxiao(imgs)
            loss = loss_fun(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum() #求每个对应位置的最大值，再与真实输入相比较，得true or false 即0和1，再求和
            total_accuracy = total_accuracy + accuracy # 整体测试集上的正确率
    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(wxiao, "wxiao_{}.pth".format(i)) # 保存模型
    # torch.save(wxiao.state_dict(), "wxiao_{}.pth.pth")

writer.close()


