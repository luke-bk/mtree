import pydicom
import numpy as np

# Load the DICOM file
file_path = '../images/dfo_images/dfo_class_0.dcm'
dicom_data = pydicom.dcmread(file_path)

# Extract the pixel array
image = dicom_data.pixel_array

# Check if the image is grayscale and 8-bit
is_grayscale = len(image.shape) == 2  # Grayscale images have 2 dimensions
is_8bit = image.dtype == np.uint8  # 8-bit images have dtype 'uint8'

print("Is Grayscale:", is_grayscale)
print("Is 8-bit:", is_8bit)

import pydicom

# Load the DICOM file
dicom_data = pydicom.dcmread(file_path)

# Access the relevant attributes
bits_allocated = dicom_data.BitsAllocated if 'BitsAllocated' in dicom_data else None
bits_stored = dicom_data.BitsStored if 'BitsStored' in dicom_data else None
high_bit = dicom_data.HighBit if 'HighBit' in dicom_data else None

print("Bits Allocated:", bits_allocated)
print("Bits Stored:", bits_stored)
print("High Bit:", high_bit)

import pydicom
import numpy as np
import cv2
import os

# Load the DICOM file
dicom_data = pydicom.dcmread(file_path)

# Extract the pixel array
image = dicom_data.pixel_array

# Normalize the image to 8-bit grayscale if it's not already
if image.dtype != np.uint8:
    image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)



import pydicom
import numpy as np
import cv2

def apply_windowing(image, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    windowed_image = np.clip(image, img_min, img_max)
    windowed_image = np.interp(windowed_image, [img_min, img_max], [0, 255])
    return windowed_image.astype(np.uint8)

# Load the DICOM file
dicom_data = pydicom.dcmread(file_path)

# Extract the pixel array and apply windowing
image = dicom_data.pixel_array
window_center, window_width = dicom_data.WindowCenter, dicom_data.WindowWidth
windowed_image = apply_windowing(image, window_center, window_width)

# Save the image as PNG
output_path = 'output_path.png'
cv2.imwrite(output_path, windowed_image)
