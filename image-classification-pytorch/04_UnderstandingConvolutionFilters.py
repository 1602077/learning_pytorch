#!/usr/bin/env python
# coding: utf-8

# ## Detecting Corners and Edges using Convolution
# cat:https://www.pexels.com/photo/city-road-street-italy-5166/
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# ### Read image

img = Image.open("datasets/images/street.jpg").convert('RGB')

# ### Convert to tensor
# * Resize the image and convert to pytorch tensor by applying the transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as tf

transforms = tf.Compose([tf.Resize(400), 
                        tf.ToTensor()])

img_tensor = transforms(img)

img_tensor

img_tensor.shape


# ### Defining filter
# * Convolution of an image with different filters can perform operations such as edge detection, blur and sharpen by applying filters.
# * Filters for sharpen, line and edge detetcion operations are below.

sharpen_kernel = [[[[0, -1, 0]], 
                   [[-1, 5, -1]], 
                   [[0, -1, 0]]]]

horizontal_line_kernel = [[[[1, 0, -1]],
                           [[0, 0, 0]],
                           [[-1, 0, 1]]]]


vertical_line_kernel = [[[[0, 1, 0]],
                         [[1, -4, 1]],
                         [[0, 1, 0]]]]


edge_detection_kernel = [[[[-1, -1, -1]],
                          [[-1, 8, -1]],
                          [[-1, -1, -1]]]]

conv_filter = torch.Tensor(sharpen_kernel) 

conv_filter.shape


# ### Applying filter
# * torch.nn.functional.conv2d accepts custom filters as opposed to torch.nn.conv2d which uses the default kernel
# * F.conv2d requires a 4d tensor as input. Hence, the unsqueeze operation

img_tensor = img_tensor.unsqueeze(0)

img_tensor.shape

conv_tensor = F.conv2d(img_tensor, conv_filter, padding=0)    

conv_tensor.shape

# ### Displaying result image
# * convert the tensor back to 3d and then to a numpy array for display

conv_img = conv_tensor[0, :, :, :]
conv_img.shape

conv_img = conv_img.numpy().squeeze()
conv_img.shape

plt.figure(figsize=(20,10))
plt.imshow(conv_img)

pool = nn.MaxPool2d(2, 2)

pool_tensor = pool(conv_tensor)

pool_tensor.shape

pool_img = pool_tensor[0, :, :, :]
pool_img.shape

pool_img = pool_img.numpy().squeeze()
pool_img.shape

plt.figure(figsize=(20,10))
plt.imshow(pool_img)

