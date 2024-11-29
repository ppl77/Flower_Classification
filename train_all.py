import torch
from torch import nn, optim
from model import model_ft, model_name, filename
from data_preprocessing import dataloaders
from train import train_model, optimizer_ft, params_to_update

#所有层训练
for param in model_ft.parameters():
    param.requires_grad = True

# 优化器和损失函数
optimizer = optim.Adam(params_to_update, lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
criterion = nn.NLLLoss()

# Load the checkpoint
checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model_ft.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])

checkpoint = torch.load(filename)


#model_ft.class_to_idx = checkpoint['mapping']
model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs  = train_model(model_ft, dataloaders, criterion, optimizer, num_epochs=10, is_inception=(model_name=="inception"), filename='serious_checkpoint.pth')