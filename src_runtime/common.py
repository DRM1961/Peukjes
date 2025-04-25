# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 09:07:15 2025

@author: dirkm
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import logging


def SetupLogger(fn, outtoconsole=False):
    Logger = logging.getLogger("mylogger")
    Logger.handlers.clear()
    
    Logger.setLevel(logging.DEBUG)
    Logger.propagate = False
    file_handler = logging.FileHandler(fn, mode="a", encoding="utf-8")    
    formatter = logging.Formatter("{asctime} - {levelname} - {message}",
                                    style="{", datefmt="%Y-%m-%d %H:%M:%S")    
    file_handler.setFormatter(formatter)
    Logger.addHandler(file_handler)

    if outtoconsole:
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        Logger.addHandler(console)        
    
    return(Logger)    

mylogger = SetupLogger("../Debug/mylogger.txt", outtoconsole=True)




class FeatureExtractor50(nn.Module):
    def __init__(self, base_model='resnet50'):
        super(FeatureExtractor50, self).__init__()
        if base_model == 'resnet50':
            model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
            self.feature_extractor = nn.Sequential(*list(model.children())[:-1])  # Remove last layer
            self.feature_dim = model.fc.in_features  # Feature size

    def forward(self, x):
        x = self.feature_extractor(x)
        return x.view(x.size(0), -1)  # Flatten

class FeatureExtractor34(nn.Module):
    def __init__(self, base_model='resnet34'):
        super(FeatureExtractor34, self).__init__()
        if base_model == 'resnet34':
            model = models.resnet34(weights='ResNet34_Weights.DEFAULT')
            self.feature_extractor = nn.Sequential(*list(model.children())[:-1])  # Remove last layer
            self.feature_dim = model.fc.in_features  # Feature size

    def forward(self, x):
        x = self.feature_extractor(x)
        return x.view(x.size(0), -1)  # Flatten

class FeatureExtractor18(nn.Module):
    def __init__(self, base_model='resnet18'):
        super(FeatureExtractor18, self).__init__()
        if base_model == 'resnet18':
            model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
            self.feature_extractor = nn.Sequential(*list(model.children())[:-1])  # Remove last layer
            self.feature_dim = model.fc.in_features  # Feature size

    def forward(self, x):
        x = self.feature_extractor(x)
        return x.view(x.size(0), -1)  # Flatten

class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # First layer (input → 128 neurons)
        self.fc2 = nn.Linear(128, 64)         # Second layer (128 → 64 neurons)
        self.fc3 = nn.Linear(64, 2)           # Output layer (64 → 2 classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # No softmax here, since CrossEntropyLoss applies it
        return x


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
