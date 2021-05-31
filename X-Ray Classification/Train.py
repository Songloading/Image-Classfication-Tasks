import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import numpy as np
from torch.autograd import Variable
from torchvision import transforms
import os
import fnmatch
from datetime import datetime
from PIL import Image
import argparse
from dataloader import X_Ray, load_raw_data

# Parameters
device = "cuda" if torch.cuda.is_available() else "cpu"
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default='E:/Northwestern/COMP_SCI 499 Project/Image-Classfication-Tasks/X-Ray Classification/archive.zip', help="the path to the zip file")
parser.add_argument("--num_epochs", type=int, default=10, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=20, help="batch size")
parser.add_argument("--learning_rate", type=float, default=3e-4, help="learning rate")
parser.add_argument("--save_model", type=bool, default=False, help="Save model?")
parser.add_argument("--mode", default='Tran', help="Train or Test")
opt = parser.parse_args()

normalize = transforms.Normalize(
    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225]
)
transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    normalize])



train_paths, train_targets, test_paths, test_targets = load_raw_data(opt.data_path)
Train_Dataset = X_Ray(train_paths, train_targets, transform, opt.data_path)
train_loader = torch.utils.data.DataLoader(Train_Dataset, batch_size = opt.batch_size, shuffle = True)
Test_Dataset = X_Ray(test_paths, test_targets, transform, opt.data_path)
test_loader = torch.utils.data.DataLoader(Test_Dataset, batch_size = opt.batch_size)


# model and optimizer
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2) #binary
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer= optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.9)

for epoch in range(opt.num_epochs):
    print('Epoch {}/{}'.format(epoch, opt.num_epochs - 1))
    print('-' * 15)
    model.train()
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        running_loss = 0.0
        running_corrects = 0

        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
    valid_loss = 0.0
    model.eval()     # Optional when not using Model Specific layer
    for data, labels in test_loader:
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()
        
        target = model(data)
        loss = criterion(target,labels)
        valid_loss = loss.item() * data.size(0)
    epoch_loss = running_loss / 110000
    epoch_acc = running_corrects.double() / 110000

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        'train', epoch_loss, epoch_acc))