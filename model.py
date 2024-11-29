import torch
from torch import nn
from torchvision import  models


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 初始化模型
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 224

    if model_name == "resnet":
        from torchvision.models import ResNet152_Weights
        weights = ResNet152_Weights.IMAGENET1K_V1 if use_pretrained else None
        model_ft = models.resnet152(weights=weights)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            nn.LogSoftmax(dim=1)
        )
    return model_ft, input_size

# 初始化 ResNet 模型
model_name = 'resnet'
feature_extract = True
model_ft, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)

# 模型移动到设备
model_ft = model_ft.to(device)
#模型保存
filename='checkpoint.pth'