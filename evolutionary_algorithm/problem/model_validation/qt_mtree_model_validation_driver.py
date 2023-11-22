import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
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
    _problem_name = "model_validation"
    _seed = 999  # Set the seed for experiment repeatability
    _is_minimization_task = True  # Set the task type
    _split_probability = 0.0  # The probability that a population will split
    _merge_threshold = 30  # The number of generations a population has to improve its score before merging

    _population_size = 48  # The population size
    _max_generations = 40  # Algorithm will terminate after this many generations
    _crossover_rate = 0.9  # Crossover rate (set between 0.0 and 1.0)
    _mutation_rate = 0.01  # Mutation rate (set between 0.0 and 1.0)

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
             results_path=_results_path)

        experiment_number += 1  # Increment the experiment counter and track this
        _seed = random_gen.randint(0, 1000)  # Set a new seed for the new experiment


if __name__ == "__main__":
    run_experiment()
