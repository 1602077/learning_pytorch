#!/usr/bin/env python
# coding: utf-8

import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets,transforms 

# ### Load data
# * Download train and test set
# * Apply transforms
# * Define dataloaders
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean, std = std)
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

data_dir = 'datasets/cifar10/train'
batch_size = 8
num_workers = 2

trainset = datasets.CIFAR10(root=data_dir,
                            train=True,
                            download=True,
                            transform=train_transform)

testset = datasets.CIFAR10(root=data_dir,
                           train=False,
                           download=True,
                           transform=test_transform)

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=num_workers)

testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=num_workers)

dataloaders = {
    'train': trainloader,
    'test': testloader
}


# ### Explore dataset
dataset_sizes = { 'train': len(trainloader), 'test': len(testloader) }
class_names = trainset.classes

print(class_names)


# ### Transfer learning
# * Load the pretrained model, Resnet18
# * Define parameters
# * The criterion to minimize in the loss function. Given this is a classification model, we will look to minimize the cross-entropy loss
# * A simple SGD optimizer with momentum which accelerate gradients vectors in the right directions and hence leads to faster converging
# * Scheduler to decay Learning Rate by a factor of 0.1 every 7 epochs
# 
from torchvision import models

model = models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
num_ftrs

model.fc = nn.Linear(num_ftrs, 10)

device = torch.device("cpu")

if torch.cuda.is_available():
    device = torch.device("cuda")
print(device)

model.to(device)

# ### Define training and test phase
# * scheduler.step() will set up the scheduler for each step in order to decay the learning rate
# * Each epoch has a training and test phase
# * model.train() will set the pre-trained model into training mode. This is only available for pre-trained models
# * running_loss will keep track of the loss at each iteration
# * running_corrects keeps a count of the number of correct predictions which will be used to calculate the accuracy of the model 
# * outputs is the list probabilities for each possible label for the batch of images (which are the inputs). We use torch.max() to get the index of the highest probability label for each image in the batch

criterion        = nn.CrossEntropyLoss()

optimizer_ft     = optim.SGD( model.parameters(), lr=0.001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR( optimizer_ft, step_size=7, gamma=0.1)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    

    for epoch in range(num_epochs):
        
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        step = 0
        
        for phase in ['train', 'test']:
            
            if phase == 'train':
                scheduler.step()
                model.train(True)  
                # Set model to training mode
                
            else:
                model.train(False)  
                # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0


            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                
                step += 1
                if step % 500 == 0:
                    print('Epoch: {} Loss: {:.4f},  Step: {}'.format(epoch, loss.item(), step))
                
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double()/ (dataset_sizes[phase]*batch_size)

            print('{} Loss: {:.4f} Acc: {:.4f} '.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
    
    print('Training complete')
    print('Best test Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model

model = train_model(model, criterion, optimizer_ft,
                         exp_lr_scheduler, num_epochs=1)


# ### Getting Predictions
def imshow(inp, title):

    inp = inp.cpu().numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    
    plt.imshow(inp)
    plt.title(title)
    plt.pause(5)  

with torch.no_grad():
    
    inputs, labels = iter(dataloaders['test']).next()
    inputs, labels = inputs.to(device), labels.to(device)
    inp = torchvision.utils.make_grid(inputs)
    
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    
    for j in range(len(inputs)):
        inp = inputs.data[j]
        imshow(inp, 'predicted:' + class_names[preds[j]])

