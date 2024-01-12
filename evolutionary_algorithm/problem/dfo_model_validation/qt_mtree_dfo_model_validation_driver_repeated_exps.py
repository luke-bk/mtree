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


# Build the VGG16 model
class VGG_net(nn.Module):
    # Defaults to VGG16. Change layers to adjust.
    def __init__(self,
                 in_channels=3,
                 num_classes=1,
                 LAYERS=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']):
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


def run_experiment():
    _problem_name = "dfo_model_validation"

    _is_minimization_task = True  # Set the task type
    _split_probability = 0.05  # The probability that a population will split
    _merge_threshold = 30  # The number of generations a population has to improve its score before merging

    _population_size = 4  # The population size
    _max_generations = 2  # Algorithm will terminate after this many generations
    _crossover_rate = 0.9  # Crossover rate (set between 0.0 and 1.0)
    _mutation_rate = 0.01  # Mutation rate (set between 0.0 and 1.0)
    _image_type = "dcm"
    # _base_image = "../../../images/dfo_images/dfo_class_0.dcm"  # The image we are evolving the counterfactual from
    _current_class = 0

    # number_experiments = 1  # Determines how many experiments we will run in a single execution
    # experiment_number = 0  # Tracks the number of experiments that have run
    _problem_name = str(_current_class) + "_dfo_model_validation"
    number_experiments = 30  # Set to run 25 experiments

    # Run the algorithm for each seed from 0 to 24
    for seed in range(number_experiments):
        _base_image = f"../../../images/test_images/{_current_class}/image_{_current_class}_{seed}.dcm"  # Dynamic base image path
        _seed = seed  # Use the loop index as the seed

        # Create an instance of the numpy random generator for experimental control
        random_gen = RandomGenerator(seed=_seed)

        # Path to where we are storing the results
        # Define the parts of the file path
        results_dir = '../../../results'
        filename = f'seed_{_seed}_qt_mtree_{_problem_name}_pop_{_population_size}_gen_{_max_generations}_cxp_{_crossover_rate}'

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


if __name__ == "__main__":
    run_experiment()
