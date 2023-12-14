import pydicom

# Load the DICOM files
original_path = '../../../images/dfo_images_trial/dfo_class_0.dcm'  # Replace with your DICOM file path
evolved_path = '../../../images/dfo_images_trial/evo.dcm'  # Replace with your DICOM file path
dicom_image_1 = pydicom.dcmread(original_path)
dicom_image_2 = pydicom.dcmread(evolved_path)

# Convert them to numpy arrays
image_1 = dicom_image_1.pixel_array
image_2 = dicom_image_2.pixel_array

import numpy as np

# Compute the absolute difference

difference = np.abs(np.int16(image_1) - np.int16(image_2))
# Make sure to clip values to the 12-bit range
difference = np.clip(difference, 0, 4095)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

plt.subplot(1, 3, 1)
plt.imshow(image_1, cmap='gray')
plt.title('Original')

plt.subplot(1, 3, 2)
plt.imshow(image_2, cmap='gray')
plt.title('Evo')

plt.subplot(1, 3, 3)
plt.imshow(difference, cmap='gray')
plt.title('Difference')

plt.show()

