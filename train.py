import torch
from torchvision import datasets, transforms, models

from train_functions import create_model
from train_functions import train_model
from train_functions import save_model
from train_functions import get_input_args

# Initiate variables with default values
arch = 'vgg16'
hidden_units = 5120
learning_rate = 0.001
epochs = 1
device = 'cpu'
data_dir = './flowers'
top_k = 1

in_arg = get_input_args()
if in_arg.arch:
    arch = in_arg.arch
if in_arg.hidden_units:
    hidden_units = in_arg.hidden_units
if in_arg.learning_rate:
    learning_rate = in_arg.learning_rate
if in_arg.epochs:
    epochs = in_arg.epochs
if in_arg.top_k:
    top_k = in_arg.top_k
if in_arg.gpu:        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
# create model        
model, criterion, optimizer, scheduler, in_features = create_model(arch, hidden_units, learning_rate)

train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])])
valid_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])])

data_dir = in_arg.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64) 

image_datasets = [train_data, valid_data]

# train model, print accuracy and loss
def train_model(model, criterion, optimizer, scheduler, epochs=2):
    model.to(device)
    
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
                top_p, top_class = ps.topk(top_k, dim=1)
                equals = top_class == labels.view(*top_class.shape) 
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Train loss: {running_loss:.3f}.. "
                  f"Valid loss: {test_loss/len(validloader):.3f}.. "
                  f"Valid accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()
    return model

model_trained = train_model(model, criterion, optimizer, scheduler, epochs) 

# save model to checkpoint
def save_model(model_trained, in_features, arch, epochs, hidden_units, optimizer):

    model_trained.cpu()
    model_trained.class_to_idx = image_datasets[0].class_to_idx
    
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
             'classifier': model_trained.classifier

             }
    
    if in_arg.save_dir:
        save_dir = in_arg.save_dir
    else:
        save_dir = 'checkpoint.pth'

    torch.save(checkpoint, save_dir) 
    
save_model(model_trained, in_features, arch, epochs, hidden_units, optimizer)

# end of the program
print('Your model has been successfully saved.')