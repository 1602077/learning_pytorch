#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# ### Load CIFAR data
# * cifar 10 - labelled subsets of a larger dataset which contains 10 categories of images
# * Load training and test datasets
# * Apply transforms
# * Define dataloaders for training and testing
mean = [0.49140126, 0.4821608,  0.44652855]
std = [0.24703369, 0.24348529, 0.26158836]

train_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, 
                         std)
])

test_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(mean, 
                         std)
])

trainset = torchvision.datasets.CIFAR10(root='datasets/cifar10/train',
                                        train=True,
                                        download=True,
                                        transform=train_transform)

testset = torchvision.datasets.CIFAR10(root='datasets/cifar10/train',
                                       train=False,
                                       download=True,
                                       transform=test_transform)

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=8,
                                          shuffle=True,
                                          num_workers=2)

testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=8,
                                         shuffle=False,
                                         num_workers=2)


# #### Configuring the neural network
# * The input size will be the channels of the images (in_size)
# * The final output will have a size equal to the number of classes for the prediction
# * The convolving kernel will have a size of k_conv_size

in_size = 3

hid1_size = 16
hid2_size = 32

out1_size = 400
out2_size = 10

k_conv_size = 5


# ### Define the Convolutional Neural Network
# 
# <b>Conv2d: </b>Applies a 2D convolution over an input signal composed of several input planes.<br>
# Parameters<br>
# in_channels (int) – Number of channels in the input image<br>
# out_channels (int) – Number of channels produced by the convolution<br>
# kernel_size (int or tuple) – Size of the convolving kernel<br>
# 
# <b>BatchNorm2d: </b>Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension) as described in the paper Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift .
# Parameters<br>
# num_features – C from an expected input of size (N,C,H,W)
# 
# <b>ReLU: </b>Activation function
# 
# <b>Maxpool2d: </b>
# Parameters:<br>
# kernel_size – the size of the window to take a max over
# 
# <b>Linear: </b>
# Parameter:<br>
# 
# in_features: 
# All the operations above used 4D Tensors of shape 
# 
# Now for fully connected layers(linear layers) we need to transform them in 1D Tensors<br>
# So to the in_features of fully connected layer we will give size
# out_features:<br>
# num_classes = number of output labels
# 
# 

#In LPPool, p=1 gives sum pooling

class ConvNet(nn.Module):
    
    def __init__(self):
        super(ConvNet, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_size, hid1_size, k_conv_size),
            nn.BatchNorm2d(hid1_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(hid1_size, hid2_size, k_conv_size),
            nn.BatchNorm2d(hid2_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        
        self.layer3 = nn.Sequential(
            nn.Linear(hid2_size * k_conv_size * k_conv_size, out1_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(out1_size, out2_size))
        
        
    def forward(self, x):
        
        out = self.layer1(x)
        out = self.layer2(out)
        
        out = out.reshape(out.size(0), -1)
        
        out = self.layer3(out)
        
        return F.log_softmax(out, dim=-1)

model = ConvNet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model.to(device)

learning_rate = 0.001
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), 
                             lr=learning_rate)


# #### Training the model and predicting accuracy

total_step = len(trainloader)
num_epochs = 10
loss_values = list()


for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(trainloader):
        
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 2000 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            
            loss_values.append(loss.item())
            
print('Finished Training')    


# ### Model Evaluation
x = (range(1, 31))

plt.figure(figsize = (12, 10))
plt.plot(x, loss_values)
plt.xlabel('Step')
plt.ylabel('Loss')

model.eval()  

with torch.no_grad():
    correct = 0
    total = 0

    for images, labels in testloader:

        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the 10000 test images: {}%'          .format(100 * correct / total))


# #### Sample Prediction

sample_img, _ = testset[23]
sample_img = np.transpose(sample_img, (1, 2, 0))
m, M = sample_img.min(), sample_img.max()
sample_img = (1/(abs(m) * M)) * sample_img + 0.5 
plt.imshow(sample_img)

test_img, test_label = testset[23]
test_img = test_img.reshape(-1,3,32,32)

out_predict = model(test_img.to(device))
_,predicted = torch.max(out_predict.data, 1)

print("Actual Label : ", test_label)
print("Predicted Label : " ,predicted.item())

