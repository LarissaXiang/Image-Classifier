import pandas as pd
import numpy as np

import torch
from torch import nn, optim
from torch.optim import lr_scheduler

import torchvision
from torchvision import datasets, transforms, models

from collections import OrderedDict
from PIL import Image
from os import listdir
import json
import argparse

from predict_functions import load_model
from predict_functions import process_image
from predict_functions import predict
from predict_functions import get_input_args

# Initiate variables with default values
checkpoint = 'ImageClassifier/checkpoint.pth'
filepath = 'ImageClassifier/cat_to_name.json'    
arch=''
image_path = 'ImageClassifier/flowers/test/100/image_07896.jpg'
topk = 5
device = 'cpu'

with open(filepath, 'r') as f:
    cat_to_name = json.load(f)

in_arg = get_input_args()
if in_arg.checkpoint:
    checkpoint = in_arg.checkpoint
if in_arg.image_path:
    image_path = in_arg.image_path
if in_arg.topk:
    topk = in_arg.topk
if in_arg.json:
    filepath = in_arg.json
if in_arg.gpu:        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   

model = load_model(checkpoint) 

def predict(image_path, model, topk=5): 
    # Use process_image function to create numpy image tensor
    pytorch_np_image = process_image(image_path)
    
    # Changing from numpy to pytorch tensor
    pytorch_tensor = torch.tensor(pytorch_np_image)
    pytorch_tensor = pytorch_tensor.float()
    
    # Removing RunTimeError for missing batch size - add batch size of 1 
    pytorch_tensor = pytorch_tensor.unsqueeze(0)
    
    # Run model in evaluation mode to make predictions 
    model.to(device)
    model.eval()
    LogSoftmax_predictions = model.forward(pytorch_tensor)
    predictions = torch.exp(LogSoftmax_predictions)
    
    # Identify top predictions and top labels
    top_p, top_class = predictions.topk(topk)
    
    
    top_p = top_p.detach().numpy().tolist()
    
    top_class = top_class.tolist()
    
    labels = pd.DataFrame({'class':pd.Series(model.class_to_idx),'flower_name':pd.Series(cat_to_name)})
    labels = labels.set_index('class')
    labels = labels.iloc[top_class[0]]
    labels['predictions'] = top_p[0]
    
    return labels

labels = predict(image_path,model,topk)
print(labels)
    
