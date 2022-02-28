from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")

img = Image.open("D:\Learning\Playing\learn_pytorch\images\90179376_abc234e5f4.jpg")
# 最好使用相对地址
print(img)

# ToTensor
# transforms 如何使用,把PIL数据和ndarray转换成tensor
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
writer.add_image("ToTensor", tensor_img)

#normalize
print(tensor_img[0][0][0])
trans_norm = transforms.Normalize([6,8,22],[8,4,1])
img_norm = trans_norm(tensor_img)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm,2)

# resize
print(img.size)
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img)
img_resize = tensor_trans(img_resize)
writer.add_image("Resize",img_resize,0)

# compose - resize -2
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, tensor_trans])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2)


writer.close()