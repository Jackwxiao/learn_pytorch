import torchvision.datasets

# train_data = torchvision.datasets.ImageNet("../dataset_image_net", train=True, transform=torchvision.transforms.ToTensor(), download=True)
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)

train_data = torchvision.datasets.CIFAR10("../dataset", train=True, transform=torchvision.transforms.ToTensor(), download=True)

vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))# 添加一个线性层   (6): Linear(in_features=4096, out_features=1000, bias=True)
print(vgg16_true)
vgg16_false.add_module('add_linear', nn.Linear(1000, 10))# 修改最后一层的输出为10，out_features=10
print(vgg16_false)
