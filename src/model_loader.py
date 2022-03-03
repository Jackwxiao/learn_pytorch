# import torch
# import torchvision
#
# model1 = torch.load("vgg16_method1.pth")
#
# model2 = torch.load("vgg16_method2.pth")
#
# vgg16 = torchvision.models.vgg16(pretrained=False)# 定义网络模型
# vgg16.load_state_dict(torch.load("vgg16_method2.pth")) # 以字典形式加载
# print(vgg16)