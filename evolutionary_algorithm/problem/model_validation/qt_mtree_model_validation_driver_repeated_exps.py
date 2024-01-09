import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn

from evolutionary_algorithm.problem.model_validation.qt_mtree_model_validation_ea import main
from helpers.random_generator import RandomGenerator
from PIL import Image
from torchvision import transforms


# 5. Build the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 2)  # Two classes: 7 and 9

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Create the model instanceee
loaded_model = Net()

# Load the saved weights
loaded_model.load_state_dict(torch.load('../../model/mnist_7_9_classifier_model_with_aug.pth'))

# Set the model to evaluation mode
loaded_model.eval()


def run_experiment():
    # Removed the static _seed assignment here as it's now dynamic within the loop
    _is_minimization_task = True
    _split_probability = 0.05
    _merge_threshold = 30

    _population_size = 512
    _max_generations = 300
    _crossover_rate = 0.9
    _mutation_rate = 0.01
    # _base_image is now dynamic, removed the static assignment
    _current_class = 7
    _problem_name = str(_current_class) + "_model_validation"

    number_experiments = 25  # Set to run 25 experiments

    # Run the algorithm for each seed from 0 to 24
    for seed in range(number_experiments):
        _base_image = f"../../../images/test_images/{_current_class}/image_{_current_class}_{seed}.png"  # Dynamic base image path
        _seed = seed  # Use the loop index as the seed

        # Create an instance of the numpy random generator with the current seed
        random_gen = RandomGenerator(seed=_seed)

        # Results directory and filename generation
        results_dir = '../../../results'
        filename = f'seed_{_seed}_qt_mtree_{_problem_name}_pop_{_population_size}_gen_{_max_generations}_cxp_{_crossover_rate}'
        _results_path = os.path.join(results_dir, filename)

        # Run the main function with the current configuration
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
             current_class=_current_class)


if __name__ == "__main__":
    run_experiment()
