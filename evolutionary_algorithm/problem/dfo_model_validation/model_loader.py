import cv2
from pydicom.encaps import encapsulate
from pydicom.uid import RLELossless
from torch.utils.data import Dataset
import os
import pydicom

from evolutionary_algorithm.evaluation.fitness_function.manhattan_distance import manhattan_distance_fitness_dcm, \
    manhattan_distance_fitness_dcm_compare
from evolutionary_algorithm.genetic_operators import MutationOperators
from helpers.random_generator import RandomGenerator

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn

import numpy as np
from PIL import Image


class DICOM_Dataset(Dataset):
    def __init__(self, image_ids, image_labels, image_base_dir, channels=3):
        self.image_ids = image_ids
        self.image_labels = image_labels
        self.image_base_dir = image_base_dir
        self.channels = channels

    def __getitem__(self, i):
        image_id = self.image_ids[i]
        label = self.image_labels[i]
        img_path = os.path.join(self.image_base_dir, image_id)

        if image_id[-4:] == '.dcm':
            dcm_image = pydicom.read_file(img_path)
            image = dcm_image.pixel_array
        else:
            img = Image.open(img_path)
            image = np.array(img)
            if len(image.shape) == 3:
                image = image[:, :, :0]  # Take only the first three channels

        # This suppresses potential division errors that may occur.
        # The resulting nan errors are dealt with in the subsequent line.
        # This could cause problems in the future, but I can't think of any at this time.
        with np.errstate(divide='ignore', invalid='ignore'):
            image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = np.nan_to_num(image, nan=0.0, posinf=1.0)

        if self.channels == 3:
            image = np.stack((image,) * 3, axis=0)

        # image = image.astype('int16') # remove this when you are normalizing the images again.
        image = torch.from_numpy(image).to(dtype=torch.float32)

        return {'image': image, 'label': label}

    def __len__(self):
        return len(self.image_ids)


# Build the model
# VGG16#
class VGG_net(nn.Module):

    def __init__(self, in_channels=3, num_classes=1,
                 LAYERS=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512,
                         'M']):  # Defaults to VGG16. Change layers to adjust.
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.conv_layers = self.create_conv_layers(LAYERS)
        self.fcs = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes))

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        for x in architecture:
            if type(x) == int:
                out_channels = x
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x


# LOADS THE MODEL FOR INFERENCE##
weights = r'../../model/dfo_vgg16_weights.pt'
loaded_model = VGG_net()  # No need to specify device here

# Update this line
x = torch.load(weights, map_location=torch.device('cpu'))

loaded_model.load_state_dict(x)
loaded_model.eval()  # Continue to use eval mode for inference
loaded_model.train(False)

import os
from torch.utils.data import DataLoader
import torch.nn.functional as F


def process_folder(folder_path, model, batch_size=60):
    image_ids = [img for img in os.listdir(folder_path) if img.endswith('.dcm')]
    image_labels = [0] * len(image_ids)  # Dummy labels

    dataset = DICOM_Dataset(image_ids=image_ids, image_labels=image_labels, image_base_dir=folder_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Keep track of the current image index across batches
    current_image_index = 0

    # Process each batch
    for batch in data_loader:
        image_tensors = batch['image']
        image_tensors = image_tensors.to('cpu')

        with torch.no_grad():
            outputs = model(image_tensors)
            # Use sigmoid for binary classification
            probabilities = torch.sigmoid(outputs).squeeze()  # Squeeze to remove any extra dimensions

        # Print or store the results
        for i in range(probabilities.size(0)):
            image_id = image_ids[current_image_index]
            probability = probabilities[i].item()  # Probability of being in the 1 class
            predicted_class = 1 if probability >= 0.5 else 0  # Class prediction based on threshold
            print(f'Image: {image_id}, Predicted class: {predicted_class}, Probability: {probability:.4f}')
            current_image_index += 1


import pydicom


def post_clean_up(original_array, evolved_array):
    evolved_array[evolved_array > 3800] = original_array[evolved_array > 3800]
    evolved_array[evolved_array < 110] = original_array[evolved_array  < 110]

    # # Image dimensions
    # image_width, image_height = 224, 224
    # patch_size = 20
    #
    # # Loop over 20 x 20 patches, moving right 20 pixels after each patch
    # for y in range(0, image_height, patch_size):  # Skip 20 pixels vertically after each row of patches
    #     # print ("Top loop")
    #     for x in range(0, image_width, patch_size):  # Skip 20 pixels horizontally
    #         # print("Inner loop")
    #         # Extract the 20x20 patches
    #         evolved_patch = evolved_array[y:y + patch_size, x:x + patch_size].copy()
    #         original_patch = original_array[y:y + patch_size, x:x + patch_size].copy()
    #
    #         # Replace the evolved patch with the original patch
    #         evolved_array[y:y + patch_size, x:x + patch_size] = original_patch
    #
    #         # Check the score (use your actual scoring function)
    #         score = manhattan_distance_fitness_dcm(loaded_model,
    #                                                evolved_array,
    #                                                original_array,
    #                                                0)
    #
    #         # Check if the score is 999 999 999 999
    #         if score < evolved_vs_original_score:
    #             # Keep the changes if score is not 999 999 999 999
    #             pass
    #         else:
    #             # Revert the changes if score is 999 999 999 999
    #             evolved_array[y:y + patch_size, x:x + patch_size] = evolved_patch



# Load the DICOM file
dicom_path = '../../../images/dfo_images_trial/dfo_class_0.dcm'  # Replace with your DICOM file path
ds = pydicom.dcmread(dicom_path)
original = pydicom.dcmread(dicom_path)
original_array = original.pixel_array

evolved_path = '../../../images/dfo_images_trial/a.dcm'  # Replace with your DICOM file path
evolved = pydicom.dcmread(evolved_path)
evolved_array = evolved.pixel_array

# Check if essential tags are present (add more if necessary)
required_tags = ['BitsAllocated', 'BitsStored', 'HighBit', 'SamplesPerPixel', 'PixelRepresentation']
for tag in required_tags:
    if tag not in ds:
        raise ValueError(f"Required tag {tag} is missing from the DICOM dataset")

# Access and copy the pixel array
pixel_array = ds.pixel_array

score = manhattan_distance_fitness_dcm(loaded_model,
                                       pixel_array,
                                       original_array,
                                       1)

original_vs_original_score = manhattan_distance_fitness_dcm(loaded_model,
                                                                    original_array,
                                                                    original_array,
                                                                    0)

evolved_vs_original_score = manhattan_distance_fitness_dcm(loaded_model,
                                                                   evolved_array,
                                                                   original_array,
                                                                   0)

print(f"Original vs Original MD: {original_vs_original_score} and Evolved vs Original MD: {evolved_vs_original_score}")

# MutationOperators.perform_gaussian_mutation_dcm_image(RandomGenerator(seed=12),
#                                                       pixel_array,
#                                                       0.99,
#                                                       0.00,
#                                                       10.1)

# def classify_image_dcm(loaded_model, image_array):
# Load and preprocess the image
# Normalize and preprocess the image array as per your existing logic
with np.errstate(divide='ignore', invalid='ignore'):
    image_array = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array))
image_array = np.nan_to_num(image_array, nan=0.0, posinf=1.0)

image_array = np.stack((image_array,) * 3, axis=0)

# Convert to torch tensor
image_tensor = torch.from_numpy(image_array).to(dtype=torch.float32).unsqueeze(0)  # Add batch dimension

# Move the tensor to the CPU
image_tensor = image_tensor.to('cpu')

# Pass the tensor through the model
with torch.no_grad():
    output = loaded_model(image_tensor)

# Use sigmoid for binary classification
probability = torch.sigmoid(output).item()
predicted_class = 1 if probability >= 0.5 else 0  # Class prediction based on threshold



# preform post clean up on the evolved image, patch by patch
post_clean_up(original_array, evolved_array)

new_fitness = manhattan_distance_fitness_dcm(loaded_model,
                                       evolved_array,
                                       original_array,
                                       0)
# print (f"New fitness {new_fitness}")

ds.PixelData = pixel_array.astype('uint16')
ds.compress(RLELossless, evolved_array)
# Save the modified DICOM file
output_path = '../../../images/dfo_images_trial/manipulated_patch_clean_up.dcm'  # Replace with your desired output path
ds.save_as(output_path)
#
# # Usage
folder_path = '../../../images/dfo_images_trial/'  # Update with your data path
process_folder(folder_path, loaded_model)
