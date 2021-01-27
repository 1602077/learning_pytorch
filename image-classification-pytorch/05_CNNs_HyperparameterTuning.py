#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

mnist_train = pd.read_csv('datasets/mnist-in-csv/mnist_train.csv')
mnist_test = pd.read_csv('datasets/mnist-in-csv/mnist_test.csv')

mnist_train.head()

mnist_train = mnist_train.dropna()
mnist_test = mnist_test.dropna()

# ### View Sample
# * We will use transpose to change the shape of image tensor<br>
# <b>.imshow()</b> needs a 2D array, or a 3D array with the third dimension being of size 3 or 4 only (For RGB or RGBA), so we will shift first axis to last<br>

random_sel = mnist_train.sample(8)
random_sel.shape

image_features = random_sel.drop('label', axis =1)
image_batch = (torch.Tensor(image_features.values / 255.)).reshape((-1, 28, 28))
image_batch.shape

grid = torchvision.utils.make_grid(image_batch.unsqueeze(1), nrow=8)
grid.shape

plt.figure (figsize = (12, 12))
plt.imshow(grid.numpy().transpose((1, 2, 0)))
plt.axis('off')


# #### Identifying features and labels

mnist_train_features = mnist_train.drop('label', axis =1)
mnist_train_target = mnist_train['label']

mnist_test_features = mnist_test.drop('label', axis =1)
mnist_test_target = mnist_test['label']


# #### converting to tensors

X_train_tensor = torch.tensor(mnist_train_features.values, dtype=torch.float)
x_test_tensor  = torch.tensor(mnist_test_features.values, dtype=torch.float) 

Y_train_tensor = torch.tensor(mnist_train_target.values, dtype=torch.long)
y_test_tensor  = torch.tensor(mnist_test_target.values, dtype=torch.long)

print(X_train_tensor.shape)
print(Y_train_tensor.shape)
print(x_test_tensor.shape)
print(y_test_tensor.shape)


# #### Reshaping the tensors according to what the CNN needs

X_train_tensor = X_train_tensor.reshape(-1, 1, 28, 28)
x_test_tensor = x_test_tensor.reshape(-1, 1, 28, 28)

print(X_train_tensor.shape)
print(Y_train_tensor.shape)
print(x_test_tensor.shape)
print(y_test_tensor.shape)


# ### Defining  CNN

import torch.nn as nn
import torch.nn.functional as F

# #### Configuring the neural network
# * The input size will be the channels of the images (in_size)
# * The final output will have a size equal to the number of classes for the prediction
# * The convolving kernel will have a size of k_conv_size

in_size = 1

hid1_size = 16 #Re-run for 32
hid2_size = 32 #Re-run for 64

out_size = 10

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
        
        self.fc = nn.Linear(512, out_size)
        
 
    def forward(self, x):
        out = self.layer1(x)
        print(out.shape)
        
        out = self.layer2(out)
        print(out.shape)
        
        out = out.reshape(out.size(0), -1)
        print(out.shape)
        
        out = self.fc(out)
        print(out.shape)
        
        return out ## F.log_softmax(out, dim=-1)

model = ConvNet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model.to(device)

X_train_tensor = X_train_tensor.to(device)
x_test_tensor  = x_test_tensor.to(device) 

Y_train_tensor = Y_train_tensor.to(device)
y_test_tensor  = y_test_tensor.to(device)

#Re-run for each different value

learning_rate = 0.001 

criterion = nn.CrossEntropyLoss() 
#criterion = nn.NLLLoss() 

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
#optimizer =torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9) 

# #### Training the model

num_epochs = 10
loss_values = list()

for epoch in range(1, num_epochs):
        
        outputs = model(X_train_tensor)
        loss = criterion(outputs,Y_train_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        print('Epoch - %d, loss - %0.5f '%(epoch, loss.item()))
        loss_values.append(loss.item())

# ### Model Evaluation

x = (range(0, 9))

plt.figure(figsize = (8, 8))
plt.plot(x, loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')

model.eval()

from sklearn.metrics import accuracy_score, precision_score, recall_score

with torch.no_grad():
    
    correct = 0
    total = 0
    
    outputs = model(x_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    
    y_test = y_test_tensor.cpu().numpy()
    predicted = predicted.cpu()
    
    print("Accuracy: ", accuracy_score(predicted, y_test))
    print("Precision: ", precision_score(predicted, y_test, average='weighted'))
    print("Recall: ", recall_score(predicted, y_test, average='weighted'))


# ### Using model for predictions 

print("sample target data = ", mnist_test_target.values[1005])

sample_img = mnist_test_features.values[1005]
sample_img = sample_img.reshape(1, 28, 28)

sample_img = sample_img[0, :, :]

plt.figure(figsize =(6, 6))
plt.imshow(sample_img)

sample = np.array(mnist_test_features.values[1005]) 

sample_tensor = torch.from_numpy(sample).float()
sample_tensor = sample_tensor.reshape(-1, 1, 28, 28)
sample_tensor = sample_tensor.to(device)

y_pred = model(sample_tensor)
y_pred

_, predicted = torch.max(y_pred.data, -1)

print (" The predicted label is : ", predicted.item())

