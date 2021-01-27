#!/usr/bin/env python
# coding: utf-8
import skimage
from skimage import data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

bird = mpimg.imread("datasets/images/bird.jpeg")
plt.title("Original Image")
plt.imshow(bird)
bird.shape
bird[200: 250, 200:250]


# ## Cleaning Transformations

# ### Reshape
bird_reshape = bird.reshape(bird.shape[0] ,-1)
bird_reshape.shape

plt.figure (figsize = (6, 6))
plt.title("Reshaped Image")
plt.imshow(bird_reshape)


# ### Resize
bird_resized = skimage.transform.resize(bird, (500, 500))
bird_resized.shape

plt.figure (figsize = (6,6))
plt.title("Resized Image")
plt.imshow(bird_resized)

aspect_ratio_original = bird.shape[1] / float(bird.shape[0])
aspect_ratio_resized = bird_resized.shape[1] / float(bird_resized.shape[0])

print("Original aspect ratio: ", aspect_ratio_original)
print("Resized aspect ratio: ", aspect_ratio_resized)


# ### Rescaling preserving aspect ratio
bird_rescaled = skimage.transform.rescale(bird_resized, (1.0, aspect_ratio_original))
bird_rescaled.shape

plt.figure(figsize=(6,6)) 
plt.title("Rescaled Image")
plt.imshow(bird_rescaled) 

aspect_ratio_rescaled = bird_rescaled.shape[1] / float(bird_rescaled.shape[0])

print("Rescaled aspect ratio: ", aspect_ratio_rescaled)


# ### Reversing color order from RGB to BGR
# Used in certain frameworks such as OpenCV and Caffe2
bird_BGR = bird[:, :, (2, 1, 0)]

plt.figure (figsize = (6, 6))
plt.title("BGR Image")
plt.imshow(bird_BGR)
bird_BGR.shape
bird_gray = skimage.color.rgb2gray(bird)

plt.figure (figsize = (6,6))
plt.title("Gray Image")
plt.imshow(bird_gray, cmap = 'gray')
bird_gray.shape


# ### Cropping 
giraffes = skimage.img_as_float(skimage.io.imread('datasets/images/giraffes.jpg')).astype(np.float32)

plt.figure (figsize = (6, 6))
plt.title("Original Image")
plt.imshow(giraffes)

giraffes.shape

def crop(image, cropx, cropy):
    y, x, c = image.shape
    
    startx = x//2 - (cropx // 8)
    starty = y//3 - (cropy // 4) 
    
    stopx = startx + cropx
    stopy = starty + 2*cropy
    
    return image[starty:stopy, startx:stopx]

giraffes_cropped = crop (giraffes, 256, 256)

plt.figure (figsize = (6,6))
plt.title("Cropped Image")
plt.imshow(giraffes_cropped)


# ### Denoising Images

from skimage.util import random_noise

sigma = 0.155
noisy_giraffes = random_noise(giraffes, var=sigma**2)

plt.figure (figsize = (6, 6))
plt.title("Image with added noise")
plt.imshow(noisy_giraffes)

from skimage.restoration import denoise_tv_chambolle, denoise_bilateral, denoise_wavelet, estimate_sigma
# #### Estimate the average noise standard deviation across color channels.
sigma_est = estimate_sigma(noisy_giraffes, 
                           multichannel=True, 
                           average_sigmas=True)
print(sigma_est)

plt.imshow(denoise_tv_chambolle(noisy_giraffes, 
                                weight=0.1, 
                                multichannel=True))

plt.imshow(denoise_bilateral(noisy_giraffes, 
                             sigma_color=0.05, 
                             sigma_spatial=15,
                             multichannel=True))

plt.imshow(denoise_wavelet(noisy_giraffes, multichannel=True))


# ## Augmentation transformations

monkeys = skimage.img_as_float(skimage.io.imread('datasets/images/monkeys.jpeg')).astype(np.float32)

plt.figure (figsize = (6, 6))
plt.title("Original Image")
plt.imshow(monkeys)

# #### Flip

monkeys_flip = np.fliplr(monkeys)

plt.figure (figsize = (6, 6))
plt.title("Horizontal Flip")
plt.imshow(monkeys_flip)

monkeys_flip = np.flipud(monkeys)

plt.figure (figsize = (6, 6))
plt.title("Vertical Flip")
plt.imshow(monkeys_flip)

mirror = skimage.img_as_float(skimage.io.imread('datasets/images/book-mirrored.jpg')).astype(np.float32)

plt.figure (figsize = (6,6))
plt.title("Original Image")
plt.imshow(mirror)

mirror_flip = np.fliplr(mirror)

plt.figure (figsize = (6, 6))
plt.title("Horizontal Flip")
plt.imshow(mirror_flip)


# #### Rotation

monkeys_rotate = np.rot90(monkeys, 3)

plt.figure (figsize = (6, 6))
plt.title("Rotated Image")
plt.imshow(monkeys_rotate)

import random
from scipy import ndarray

def random_rotation(image_array: ndarray):

    random_degree = random.uniform(-25, 25)
    return skimage.transform.rotate(image_array, random_degree)

monkeys_angle= random_rotation (monkeys)

plt.figure (figsize = (6,6))
plt.title("Rotated Image")
plt.imshow(monkeys_angle)


# #### Swirl

from skimage.transform import swirl
monkeys_swirl = swirl(monkeys, strength=10, radius = 210)

plt.figure (figsize = (6, 6))
plt.title("Swirled Image")
plt.imshow(monkeys_swirl)

