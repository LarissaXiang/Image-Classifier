import pandas as pd
import numpy as np

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms

import torchvision
from torchvision import datasets, transforms, models

from collections import OrderedDict
from os import listdir
import time
import copy
import argparse


def create_model(arch, hidden_units, learning_rate):
    model = getattr(models,arch)(pretrained=True)
    in_features = model.classifier[0].in_features        
        
    for param in model.parameters():
        param.requires_grad = False
      
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(in_features, hidden_units)),
                          ('drop', nn.Dropout(p=0.5)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier = classifier
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(
        model.classifier.parameters(),lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer,step_size=4,gamma=0.1,last_epoch=-1)
    
    return model, criterion, optimizer, scheduler
    
def train_model(model, criterion, optimizer, scheduler, epochs=2):

    epochs = epochs
    steps = 0
    running_loss = 0
    print_every = 10

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
        # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_loss = running_loss*inputs.shape[0]        
        
        test_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)
                    
                test_loss += batch_loss.item()
                    
                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(topk, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Train loss: {running_loss:.3f}.. "
                  f"Valid loss: {test_loss/len(validloader):.3f}.. "
                  f"Valid accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()
    return model

def save_model(model_trained):

    model_trained.class_to_idx = image_datasets['train'].class_to_idx
    model_trained.cpu()
    save_dir = ''
    checkpoint = {
             'input_size': in_features,
             'output_size': 102,
             'arch': arch,
             'learning_rate': 0.01,
             'batch_size': 64,
             'epochs': epochs,
             'hidden_units': hidden_units, 
             'optimizer': optimizer.state_dict(),
             'state_dict': model_trained.state_dict(),
             'class_to_idx': model_trained.class_to_idx,
             }
    
    if args.save_dir:
        save_dir = args.save_dir
    else:
        save_dir = 'checkpoint.pth'

    torch.save(checkpoint, save_dir) 
    
def get_input_args():
    parser = argparse.ArgumentParser(prog='train')
    
    parser.add_argument('--data_dir',type=str, help='Location of directory with data for image classifier to train and test', default = './ImageClassifier/flowers')
    parser.add_argument('--save_dir', type=str, default='./ImageClassifier')
    parser.add_argument('--path_to_image', type=str, default='./ImageClassifier/flowers')
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--arch', type=str, default='vgg13')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--hidden_units', type=int, default=512)
    parser.add_argument('--top_k', type=int, default=3)
    
    return parser.parse_args()