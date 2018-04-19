#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
ZetCode PyQt5 tutorial 

This program creates a menubar. The
menubar has one menu with an exit action.

Author: Jan Bodnar
Website: zetcode.com 
Last edited: January 2017
"""

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import math
import torch.utils.model_zoo as model_zoo


from PIL import Image
import PIL
import torchvision.transforms
import torchvision.models
from torchvision.models.resnet import model_urls
#imports pour faire le bouton pour charger les images utilisateurs
import tkinter
from tkinter import filedialog
from ipywidgets import widgets
import sys
from PyQt5.QtWidgets import QMainWindow, QLabel, QAction, qApp, QApplication,QScrollArea, QFileDialog, QGridLayout, QWidget, QHBoxLayout
from PyQt5.QtGui import QIcon, QPixmap
plt.ion()   # interactive mode


__all__ = ['ResNet', 'resnet18']

class_names = ['A','D1','D2','D21','D28','D35','D36','D4','D42','D46','D54','D56','D58','D60','E1','E23','E34','E9','F13','F16','F31','F34','G1','G14','G17','G25','G35','G36','G39','G4','G40','G43','G5','G7','H6','I10','I9','L1','L2','M17','M18','M23','N1','N14','N17','N18','N29','N30','N31','N33','N35','N37','N5','O1','O22','O23','O28','O34','O4','O49','O50','P8','Q1','Q3','R8','S29','S34','T22','U1','U15','U33','V13','V28','V30','V31','V4','V7','W18','W19','W24','W25','W9','X1','X8','Y2','Y5','Z1','Z11']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropRate=0.5):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.downsample = downsample
        self.stride = stride
        self.droprate = dropRate

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


    
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    data_dir = ".\\train_vega"
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                 shuffle=True, num_workers=4)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    # récupère les poids à chaque itération
                    # print(torch.sum(model.fc.weight.data))

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            if phase == 'val':
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)
                
            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model, 'vega_model.pt')
    return model


class Example(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def save(self):    
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        torch.save(model_ft.state_dict(), file+'/model.pt')
        print('done')
            

    def load(self, e):
        global model_ft
        root = tkinter.Tk()
        root.withdraw()
        currdir = os.getcwd()
        model_path = filedialog.askopenfilename(parent=root, initialdir=currdir, title='Please select a model')
        model_ft = torch.load(model_path,map_location=lambda storage, loc : storage)
        print('done')
        
    def empty(self, e):
        for i in reversed(range(self.gridlayout.count())): 
            self.gridlayout.itemAt(i).widget().setParent(None)
        
    def prep(self,e):
        global model_ft
        model_ft = resnet18(pretrained=True)
        for param in model_ft.parameters():
            param.requires_grad = True
        
        num_ftrs = model_ft.fc.in_features
        model_ft.fc1 = nn.Linear(num_ftrs, 512)
        model_ft.fc2 = nn.Linear(model_ft.fc1.in_features, 256)
        model_ft.fc = nn.Linear(model_ft.fc2.in_features, 88) #Add a layer
        
        criterion = nn.CrossEntropyLoss()
        
        # Observe that all parameters are being optimized
        
        #SGD
        #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0001)
        
        #Adam
        optimizer_ft = optim.Adam(model_ft.parameters(), lr = 0.0001)
        
        #Adagrad
        #optimizer_ft = optim.Adagrad(model_ft.parameters(), lr = 0.0001)
        
        #RMSProp
        #optimizer_ft = optim.RMSprop(model_ft.parameters(), lr = 0.0001)
        
        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        return train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=5)

    
    def initUI(self):
        self.statusBar()
        menubar = self.menuBar()
        train = QAction('&Train Model', self)  
        train.triggered.connect(self.prep)
        save = QAction('&Save Model', self)
        save.triggered.connect(self.save)
        load = QAction('&Load Model', self)
        load.triggered.connect(self.load)
        empty = QAction('&Clear window', self)  
        empty.triggered.connect(self.empty)
        menubar.addAction(train)
        menubar.addAction(save)
        menubar.addAction(load)
        #menubar.addAction(load)
        self.setGeometry(200, 200, 800, 800)
        self.setWindowTitle('VEGA')
        self.grid = grid(parent=self)
        self.setCentralWidget(self.grid)

        
class grid(QWidget):
    

    def __init__(self,parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.gridlayout = QGridLayout(self.scrollAreaWidgetContents)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.layout.addWidget(self.scrollArea)
        
        self.setAcceptDrops(True)
        
    def image_loader(self,image_name):
        imsize = 256
        loader = transforms.Compose([
        #transforms.Grayscale(),
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
        """load image, returns cuda tensor"""
        image = Image.open(image_name).convert('RGB')
        image = loader(image).float()
        image = Variable(image, requires_grad=True)
        image = image.unsqueeze(0)
        return image
        
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        super().__init__()
        global lin
        global col
        for url in e.mimeData().urls():
            print(url.toLocalFile())
            pixmap = QPixmap(url.toLocalFile())
            pixmap = pixmap.scaled(100, 100)
            label = QLabel(self)
            label.setPixmap(pixmap)
            self.gridlayout.addWidget(label, lin, col)
            label = QLabel(self)
            
            image = self.image_loader(url.toLocalFile())
            model_ft(image)
    
            outputs = model_ft(image)
            _, preds = torch.max(outputs.data, 1)
            
            print(class_names[preds[0]])
            
            
            label.setText('Predict : '+class_names[preds[0]])
            self.gridlayout.addWidget(label, lin, col+1)
            if col < 4:
                col = col + 2
            else:
                col = 0
                lin = lin + 1
        
if __name__ == '__main__':    
    app = QApplication(sys.argv)
    lin=0
    col=0
    ex = Example()
    ex.show()
    sys.exit(app.exec_())