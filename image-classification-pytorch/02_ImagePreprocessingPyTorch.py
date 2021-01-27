#!/usr/bin/env python
# coding: utf-8

# ## Mean Normalization 

import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('datasets/cifar-10-batches-py/data_batch_1', 'rb') as input_file: 
    X = pickle.load(input_file, encoding='latin1')

with open('datasets/cifar-10-batches-py/data_batch_1', 'rb') as input_file: 
    X = pickle.load(input_file, encoding='latin1')
X.keys()
X = X['data']
X.shape

X = X.reshape((-1, 3, 32, 32))
X.shape

X = X.transpose(0, 2, 3, 1)
X.shape

X = X.reshape(-1, 3 * 32 * 32) 
X.shape

plt.imshow(X[6].reshape(32, 32, 3))
plt.show()


# ## Normalization
# * zero-centre the data :this calculates the mean separately across pixels and colour channels
# * divide by std

X = X - X.mean(axis=0)
X = X / np.std(X, axis=0) 

def show(i):
    i = i.reshape((32, 32, 3))

    m, M = i.min(), i.max()
    
    plt.imshow((i - m) / (M - m))
    plt.show()

show(X[6])


# ### ZCA whitening
# Whitening is a transformation of data in such a way that its covariance matrix Σ is the identity matrix. Hence whitening decorrelates features. It is used as a preprocessing method.  Principal component analysis (PCA) and Zero-phase component analysis (ZCA) are the two ways to do this.
# 
# * compute the covariance of the image data
# * perform singular value decomposition (These steps take time)
# * build the ZCA matrix
# * transform the image data  
X_subset = X[:1000]

X_subset.shape
cov = np.cov(X_subset, rowvar=True)   
cov.shape

U, S, V = np.linalg.svd(cov)     
print(U.shape)
print(S.shape)
print(V.shape)
epsilon = 1e-5

zca_matrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))

zca_matrix.shape
zca = np.dot(zca_matrix, X_subset)   
zca.shape
show(zca[6])


# ## Pre-processing in PyTorch
import torch
import torchvision
import torchvision.transforms as transforms

dir(transforms)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor()
])

dataset = torchvision.datasets.CIFAR10(root='./datasets/cifar10/train', download=True, transform=transform)
dataset
dataloader = torch.utils.data.DataLoader(dataset, 
                                         batch_size=16,
                                         shuffle=True, 
                                         num_workers=2)


# ### Viewing the images

images_batch, labels_batch = iter(dataloader).next()

images_batch.shape
labels_batch.shape
labels_batch

img = torchvision.utils.make_grid(images_batch)
img.shape
img = np.transpose(img, (1, 2, 0))
img.shape

plt.figure (figsize = (16, 12))
plt.imshow(img)
plt.axis('off')
plt.show()


# ## Normalization
# 
# Data normalization is an important step which ensures that each input parameter (pixel, in this case) has a similar data distribution. This makes convergence faster while training the network. Data normalization is done by subtracting the mean from each pixel and then dividing the result by the standard deviation
# 
# * Finding mean and std

pop_mean = []
pop_std = []

for i, data in enumerate(dataloader, 0):
    
    # shape (batch_size, 3, height, width)
    numpy_image = data[0].numpy() 
    
    # shape (3,)
    batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
    batch_std = np.std(numpy_image, axis=(0, 2, 3))
    
    pop_mean.append(batch_mean)
    pop_std.append(batch_std)

pop_mean = np.array(pop_mean)
pop_std = np.array(pop_std)

pop_mean.shape, pop_std.shape
# shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)

pop_mean = pop_mean.mean(axis=0)
pop_std = pop_std.mean(axis=0)

print(pop_mean)
print(pop_std)


# ### Applying transforms 
transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(pop_mean, 
                                 pop_std)
            ])

trainset = torchvision.datasets.CIFAR10(root='./datasets/cifar10/train', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                          shuffle=True, num_workers=2)

images_batch, labels_batch = iter(trainloader).next()

images_batch.shape


# ### Viewing the images
img = torchvision.utils.make_grid(images_batch)

img.shape
img = np.transpose(img, (1, 2, 0))

img.shape

m, M = img.min(), img.max()

m, M
# Ensure floating point image RGB values must be in the 0..1 range.

img = (1/(abs(m) * M)) * img + 0.5 
img
plt.figure (figsize = (16, 12))

plt.imshow(img)
plt.show()


