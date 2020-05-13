import ast
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
from torchvision import datasets
from torch import __version__
import os
import json
import torch

import argparse

def load_model(checkpoint):
    checkpoint = torch.load(checkpoint)
    
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        in_features = 25088
        for param in model.parameters():
            param.requires_grad = False
    elif checkpoint['arch'] == 'alexnet':
        model = models.alexnet(pretrained=True)
        in_features = 9216
        for param in model.parameters():
            param.requires_grad = False
    elif checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
        in_features = 1024
        for param in model.parameters():
            param.requires_grad = False
    else:
        print('The required architecture cannot be recognised')
    
    model.class_to_idx = checkpoint['class_to_idx']
    hidden_units = checkpoint['hidden_units']
    
    classifier = nn.Sequential(OrderedDict([
                         ('fc1',nn.Linear(in_features,hidden_units)),
                           ('ReLu1',nn.ReLU()),
                           ('Dropout1',nn.Dropout(p=0.15)),
                           ('fc2',nn.Linear(hidden_units,512)),
                           ('ReLu2',nn.ReLU()),
                           ('Dropout2',nn.Dropout(p=0.15)),
                           ('fc3',nn.Linear(512,102)),
                           ('output',nn.LogSoftmax(dim=1))
                           ]))    
    
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def process_image(image_path):
    
    # Process a PIL image for use in a PyTorch model
    size = 256, 256
    crop_size = 224
    
    im = Image.open(image_path)
    
    im.thumbnail(size)

    left = (size[0] - crop_size)/2
    top = (size[1] - crop_size)/2
    right = (left + crop_size)
    bottom = (top + crop_size)

    im = im.crop((left, top, right, bottom))
    
    np_image = np.array(im)
    np_image = np_image/255
    
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    
    np_image = (np_image - means) / stds
    pytorch_np_image = np_image.transpose(2,0,1)
    
    return pytorch_np_image

def predict(image_path, model, topk=5): 
    # Use process_image function to create numpy image tensor
    pytorch_np_image = process_image(image_path)
    
    # Changing from numpy to pytorch tensor
    pytorch_tensor = torch.tensor(pytorch_np_image)
    pytorch_tensor = pytorch_tensor.float()
    
    # Removing RunTimeError for missing batch size - add batch size of 1 
    pytorch_tensor = pytorch_tensor.unsqueeze(0)
    
    # Run model in evaluation mode to make predictions
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

def get_input_args():
    parser = argparse.ArgumentParser(prog='predict')
    
    parser.add_argument('--checkpoint', type=str, help='Name of trained model to be loaded and used for predictions.')
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--image_path', type=str, default='./ImageClassifier/flowers/test/1/image_06743.jpg')
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--json', type=str)
    
    return parser.parse_args()