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
import os
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import pydicom


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
            probability = probabilities[i].item()  # Probability of being in the positive class
            predicted_class = 1 if probability >= 0.5 else 0  # Class prediction based on threshold
            print(f'Image: {image_id}, Predicted class: {predicted_class}, Probability: {probability:.4f}')
            current_image_index += 1


# Load the DICOM file
dicom_path = '../../../images/dfo_images_trial/dfo_class_0.dcm'  # Replace with your DICOM file path
original_dicom_image_path = pydicom.dcmread(dicom_path)
original_dicom_image = pydicom.dcmread(dicom_path)
original_dicom_as_array = original_dicom_image.pixel_array
to_mutate_dicom_as_array = original_dicom_image.pixel_array.copy()

# Check if essential tags are present (add more if necessary)
required_tags = ['BitsAllocated', 'BitsStored', 'HighBit', 'SamplesPerPixel', 'PixelRepresentation']
for tag in required_tags:
    if tag not in original_dicom_image_path:
        raise ValueError(f"Required tag {tag} is missing from the DICOM dataset")

original_vs_original_score = manhattan_distance_fitness_dcm(loaded_model,
                                                            original_dicom_as_array,
                                                            original_dicom_as_array,
                                                            1) # hack set to class one, so it is not penalised

print(f"Original vs Original MD: {original_vs_original_score}")

random_gen = RandomGenerator(seed=2)
# Preform mutation
for x in range(200):
    MutationOperators.perform_gaussian_mutation_dcm_patch(random_gen,
                                                          to_mutate_dicom_as_array,
                                                          0.5,
                                                          0.00,
                                                          1000.1)


MutationOperators.replace_patch_from_original(random_gen,
                                                original_dicom_as_array,
                                                to_mutate_dicom_as_array)

mutated_vs_original_score = manhattan_distance_fitness_dcm(loaded_model,
                                                           to_mutate_dicom_as_array,
                                                           original_dicom_as_array,
                                                           0)

print(f"Mutated vs Original MD: {mutated_vs_original_score}")


# Save out image for testing
original_dicom_image_path.PixelData = to_mutate_dicom_as_array.astype('uint16')
original_dicom_image_path.compress(RLELossless, to_mutate_dicom_as_array)
# Save the modified DICOM file
output_path = '../../../images/dfo_images_trial/patch_noise.dcm'  # Replace with your desired output path
original_dicom_image_path.save_as(output_path)
#
# # Usage
folder_path = '../../../images/dfo_images_trial/'  # Update with your folder path
process_folder(folder_path, loaded_model)
