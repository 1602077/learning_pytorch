#!/usr/bin/env python
# coding: utf-8

# # Building a dense, fully-connected neural network for image classification
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ### Load and explore data
mnist_train = pd.read_csv('datasets/mnist-in-csv/mnist_train.csv')
mnist_test = pd.read_csv('datasets/mnist-in-csv/mnist_test.csv')

mnist_train.shape, mnist_test.shape
mnist_train.head()

# #### display an image
img = mnist_train[1:2]
img = img.drop('label', axis =1)

img = img.values
img.shape

img = img.reshape(1, 28, 28)
img.shape

img = img.squeeze()
img.shape

plt.figure(figsize = (6, 6))
plt.imshow(img)

mnist_test.head()

# ### Data Preprocessing
mnist_train = mnist_train.dropna()
mnist_test = mnist_test.dropna()

# #### Identifying features and labels
mnist_train_features = mnist_train.drop('label', axis =1)
mnist_train_target = mnist_train['label']

mnist_test_features = mnist_test.drop('label', axis =1)
mnist_test_target = mnist_test['label']

mnist_train_features.head()

mnist_test_features.head()

mnist_train_target.head()

mnist_test_target.head()


# #### Normalization
print("train max - ", mnist_train.values.max())
print("train min - ", mnist_train.values.min())
print("test max - ", mnist_test.values.max())
print("test min - ", mnist_test.values.min())

mnist_train = mnist_train.astype('float32')
mnist_train = mnist_train/255

mnist_test = mnist_test.astype('float32')
mnist_test = mnist_test/255

print("train max - ", mnist_train.values.max())
print("train min - ", mnist_train.values.min())
print("test max - ", mnist_test.values.max())
print("test min - ", mnist_test.values.min())


# #### converting to tensors
# * Labels are in one column. Converting that one row but multiple columns (for loss func, target should be a 1D tensor; y vector should be of type long)
import torch

X_train_tensor = torch.tensor(mnist_train_features.values, dtype=torch.float)
x_test_tensor  = torch.tensor(mnist_test_features.values, dtype=torch.float) 

Y_train_tensor = torch.tensor(mnist_train_target.values, dtype=torch.long)
y_test_tensor  = torch.tensor(mnist_test_target.values, dtype=torch.long)

X_train_tensor.shape, Y_train_tensor.shape
x_test_tensor.shape, y_test_tensor.shape


# ### Defining DNN model and parameters
import torch.nn as nn

input_size = 784
output_size = 10

hidden1_size = 16
hidden2_size = 32

class Net(nn.Module):
    
    def __init__(self):
        
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size) 
        self.fc2 = nn.Linear(hidden1_size, hidden2_size) 
        self.fc3 = nn.Linear(hidden2_size, output_size) 
    
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        
        x = self.fc3(x)
        
        return torch.log_softmax(x, dim=-1)

model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model.to(device)

X_train_tensor = X_train_tensor.to(device)
x_test_tensor  = x_test_tensor.to(device) 

Y_train_tensor = Y_train_tensor.to(device)
y_test_tensor  = y_test_tensor.to(device)

import torch.optim as optim

optimizer = optim.Adam(model.parameters())

loss_fn = nn.NLLLoss()


# ### training
epochs = 500

for epoch in range(1, epochs + 1):

    optimizer.zero_grad()
    Y_pred = model(X_train_tensor)

    loss = loss_fn(Y_pred , Y_train_tensor)
    loss.backward()

    optimizer.step()
    
    if epoch % 10 == 0:
        print('Epoch - %d, loss - %0.2f ' %(epoch, loss.item()))


# ### Model Evaluation
# * Model.eval tells the network that it is in testing/evaluation phase. Dropout and batch normalisation, in particular, behave differently during testing and training and this will tell it which behaviour to adopt for the following run.

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

sample_img = sample_img[0,:,:]

plt.figure(figsize =(6, 6))
plt.imshow(sample_img)

sample = np.array(mnist_test_features.values[1005]) 

sample_tensor = torch.from_numpy(sample).float()

y_pred = model(sample_tensor.to(device))
y_pred

_, predicted = torch.max(y_pred.data, -1)

print (" The predicted label is : ", predicted.item())

