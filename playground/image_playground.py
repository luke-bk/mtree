import pydicom
from pydicom.encaps import encapsulate
import numpy as np
import os
#
# # Function to decompress the image
# def decompress_image(dicom_dataset):
#     # Check if GDCM or Pillow is available for decompression
#     # Decompress the image
#     # Return the decompressed image
#     pass
#
# # Function to recompress the image
# def recompress_image(image):
#     # Use appropriate library to recompress the image
#     # Return the recompressed image
#     pass
#
# # Load the DICOM file
# file_path = '../images/dfo_images/dfo_class_0.dcm'
# dicom_data = pydicom.dcmread(file_path)
#
# # Decompress the pixel data
# image = decompress_image(dicom_data)
#
# # Modify a pixel value (example: change the pixel at row 10, column 15)
# row, col = 10, 15
# new_value = 128
# image[row, col] = new_value
#
# # Recompress the modified image
# recompressed_image = recompress_image(image)
#
# # Encapsulate the recompressed pixel data
# dicom_data.PixelData = encapsulate([recompressed_image])
#
# # Save the modified DICOM file
# output_directory = '../images/dfo_images/'
# output_file = 'modified_compressed_image.dcm'
# output_path = os.path.join(output_directory, output_file)
#
# pydicom.dcmwrite(output_path, dicom_data)
# print(f"Saved modified DICOM file at {output_path}")





# Load the DICOM file
file_path = '../images/dfo_images/dfo_class_0.dcm'
dicom_data = pydicom.dcmread(file_path)

# Get the TransferSyntaxUID
transfer_syntax = dicom_data.file_meta.TransferSyntaxUID

# Print the TransferSyntaxUID
print(f"TransferSyntaxUID: {transfer_syntax}")

# Determine the compression type based on the TransferSyntaxUID
if transfer_syntax.is_compressed:
    print("The DICOM file is compressed.")
    if transfer_syntax == pydicom.uid.JPEGBaseline:
        print("Compression Type: JPEG Baseline (Process 1)")
    elif transfer_syntax == pydicom.uid.JPEGExtended:
        print("Compression Type: JPEG Extended (Process 2 & 4)")
    elif transfer_syntax == pydicom.uid.JPEGLossless:
        print("Compression Type: JPEG Lossless")
    elif transfer_syntax == pydicom.uid.JPEGLosslessSV1:
        print("Compression Type: JPEG Lossless, SV1")
    elif transfer_syntax == pydicom.uid.JPEG2000Lossless:
        print("Compression Type: JPEG 2000 Lossless")
    elif transfer_syntax == pydicom.uid.JPEG2000:
        print("Compression Type: JPEG 2000")
    elif transfer_syntax == pydicom.uid.RLELossless:
        print("Compression Type: RLE Lossless")
    # Add more conditions for other compression types as needed
else:
    print("The DICOM file is not compressed.")










# import pydicom
#
# # Load the DICOM file
# dicom_data = pydicom.dcmread(file_path)
#
# # Access the relevant attributes
# bits_allocated = dicom_data.BitsAllocated if 'BitsAllocated' in dicom_data else None
# bits_stored = dicom_data.BitsStored if 'BitsStored' in dicom_data else None
# high_bit = dicom_data.HighBit if 'HighBit' in dicom_data else None
#
# print("Bits Allocated:", bits_allocated)
# print("Bits Stored:", bits_stored)
# print("High Bit:", high_bit)
# #
# import pydicom
# import numpy as np
# import cv2
# import os
#
# # Load the DICOM file
# dicom_data = pydicom.dcmread(file_path)
#
# # Extract the pixel array
# image = dicom_data.pixel_array
#
# # Normalize the image to 8-bit grayscale if it's not already
# if image.dtype != np.uint8:
#     image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)
#
#
#
# import pydicom
# import numpy as np
# import cv2
#
# def apply_windowing(image, window_center, window_width):
#     img_min = window_center - window_width // 2
#     img_max = window_center + window_width // 2
#     windowed_image = np.clip(image, img_min, img_max)
#     windowed_image = np.interp(windowed_image, [img_min, img_max], [0, 255])
#     return windowed_image.astype(np.uint8)
#
# # Load the DICOM file
# dicom_data = pydicom.dcmread(file_path)
#
# # Extract the pixel array and apply windowing
# image = dicom_data.pixel_array
# # window_center, window_width = dicom_data.WindowCenter, dicom_data.WindowWidth
# # windowed_image = apply_windowing(image, window_center, window_width)
#
# # Find the pixel range
# pixel_min = np.min(image)
# pixel_max = np.max(image)
# print(f"Pixel Range: {pixel_min} to {pixel_max}")
#
# import pydicom
#
# # Load the DICOM file
# dicom_data = pydicom.dcmread(file_path)
#
# # Get the relevant header information
# bits_stored = dicom_data[0x0028, 0x0101].value  # Bits Stored
# pixel_representation = dicom_data[0x0028, 0x0103].value  # Pixel Representation
#
# # Print the information
# print(f"Bits Stored: {bits_stored}")
# print(f"Pixel Representation: {'Signed' if pixel_representation == 1 else 'Unsigned'}")
#
# # Decompress the image if necessary (using an appropriate library)
# if dicom_data.file_meta.TransferSyntaxUID.is_compressed:
#     print ("Compressed")
# else:
#     print("Not compressed")
#
# # # Save the image as PNG
# # output_path = 'output_path.png'
# # cv2.imwrite(output_path, windowed_image)
