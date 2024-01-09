import numpy as np
import idx2numpy
import cv2

# File paths for the MNIST dataset
image_file = '../../../data/MNIST/raw/train-images-idx3-ubyte'
label_file = '../../../data/MNIST/raw/train-labels-idx1-ubyte'

# Load the images and labels
images = idx2numpy.convert_from_file(image_file)
labels = idx2numpy.convert_from_file(label_file)

# Find the indices of images labeled as '7' and '9'
indices_7 = np.where(labels == 7)[0][50:70]
indices_9 = np.where(labels == 9)[0][50:70]

# Extract images for '7' and '9'
images_7 = images[indices_7]
images_9 = images[indices_9]

# Save the images to disk
for idx, image in enumerate(images_7):
    cv2.imwrite(f'image_7_{idx}.png', image)

for idx, image in enumerate(images_9):
    cv2.imwrite(f'image_9_{idx}.png', image)
