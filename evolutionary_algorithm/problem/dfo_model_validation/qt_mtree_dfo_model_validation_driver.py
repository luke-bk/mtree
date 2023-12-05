import os

import pydicom

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn

from torchvision import transforms
import numpy as np
from PIL import Image

from evolutionary_algorithm.problem.dfo_model_validation.qt_mtree_dfo_model_validation_ea import main
from helpers.random_generator import RandomGenerator


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


def classify_image_dcm(loaded_model, image_array):
    # Load and preprocess the image
    # Normalize and preprocess the image array as per your existing logic
    with np.errstate(divide='ignore', invalid='ignore'):
        image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
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

    # Return the predicted class and probability
    return predicted_class, probability


def loop_folder(loaded_model, folder_path):
    # List all files in the directory
    for filename in os.listdir(folder_path):
        # Construct the full file path
        file_path = os.path.join(folder_path, filename)

        # Check if it's a file and not a directory
        if os.path.isfile(file_path):
            # Load the image
            try:
                dicom_data = pydicom.dcmread(file_path)
                image_array = dicom_data.pixel_array

                # Classify the image
                class_name, probability = classify_image_dcm(loaded_model, image_array)

                # Print the results
                print(f"Image: {filename}, Classification: {class_name}, Confidence: {probability:.2f}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")


def run_experiment():
    _problem_name = "dfo_model_validation"
    _seed = 1  # Set the seed for experiment repeatability
    _is_minimization_task = True  # Set the task type
    _split_probability = 0.05  # The probability that a population will split
    _merge_threshold = 30  # The number of generations a population has to improve its score before merging

    _population_size = 10  # The population size
    _max_generations = 1  # Algorithm will terminate after this many generations
    _crossover_rate = 0.9  # Crossover rate (set between 0.0 and 1.0)
    _mutation_rate = 0.01  # Mutation rate (set between 0.0 and 1.0)
    _image_type = "dcm"
    _base_image = "../../../images/dfo_images/dfo_class_0.dcm"  # The image we are evolving the counterfactual from
    _current_class = 0

    number_experiments = 1  # Determines how many experiments we will run in a single execution
    experiment_number = 0  # Tracks the number of experiments that have run

    # Run the algorithm for
    while experiment_number < number_experiments:
        # Create an instance of the numpy random generator for experimental control
        random_gen = RandomGenerator(seed=_seed)

        # Path to where we are storing the results
        # Define the parts of the file path
        results_dir = '../../../results'
        filename = f'qt_mtree_{_problem_name}_seed_{_seed}_pop_{_population_size}_gen_{_max_generations}_cxp_{_crossover_rate}'

        # Construct the full file path
        _results_path = os.path.join(results_dir, filename)

        main(loaded_model,
             random_gen,
             is_minimization_task=_is_minimization_task,
             split_probability=_split_probability,
             merge_threshold=_merge_threshold,
             population_size=_population_size,
             max_generations=_max_generations,
             crossover_rate=_crossover_rate,
             mutation_rate=_mutation_rate,
             results_path=_results_path,
             base_image=_base_image,
             image_type=_image_type,
             current_class=_current_class)

        experiment_number += 1  # Increment the experiment counter and track this
        _seed = random_gen.randint(0, 1000)  # Set a new seed for the new experiment


if __name__ == "__main__":
    # run_experiment()
    # Usage
    image_locations = "../../../images/test_dfo_sample/"
    loop_folder(loaded_model, image_locations)
