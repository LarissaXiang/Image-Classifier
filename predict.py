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

checkpoint = 'checkpoint.pth'
filepath = './cat_to_name.json'    
arch=''
image_path = './flowers/test/100/image_07896.jpg'
topk = 5

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
    
with open(filepath, 'r') as f:
    cat_to_name = json.load(f)

model = load_model(checkpoint) 

labels = predict(image_path,model,topk)
print(labels)
    
