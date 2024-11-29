import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from model import initialize_model, model_name, feature_extract, device
from  data_preprocessing import dataloaders, label
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#网络效果测试
model_ft, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)
# GPU模式
model_ft = model_ft.to(device)
#保存文件的名字
filename='serious_checkpoint.pth'
# 加载模型
checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model_ft.load_state_dict(checkpoint['state_dict'])

#测试数据预处理
def process_image(image_path):
    # 读取测试数据
    img = Image.open(image_path)
    # Resize,thumbnail方法只能进行缩小，所以进行了判断
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
    # Crop操作
    left_margin = (img.width - 224) / 2
    bottom_margin = (img.height - 224) / 2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin,
                    top_margin))
    # 相同的预处理方法
    img = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])  # provided mean
    std = np.array([0.229, 0.224, 0.225])  # provided std
    img = (img - mean) / std

    # 注意颜色通道应该放在第一个位置
    img = img.transpose((2, 0, 1))

    return img

# 得到一个batch的测试数据
dataiter = iter(dataloaders['valid'])
images, labels = next(dataiter)
model_ft.eval()
output = model_ft(images.cuda())

_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.cpu().numpy())


fig=plt.figure(figsize=(20, 20))
columns =4
rows = 2

def im_convert(tensor):

    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image

for idx in range (columns*rows):
    ax = fig.add_subplot(rows, columns, idx+1, xticks=[], yticks=[])
    plt.imshow(im_convert(images[idx]))
    ax.set_title("{} ({})".format(label[str(preds[idx])], label[str(labels[idx].item())]),
                 color=("green" if label[str(preds[idx])]==label[str(labels[idx].item())] else "red"))
plt.show()