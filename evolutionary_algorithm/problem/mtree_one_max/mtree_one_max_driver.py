import os

from evolutionary_algorithm.problem.mtree_one_max.mtree_one_max_ea import main
from helpers.random_generator import RandomGenerator

_seed = 5  # Set the seed for experiment repeatability
_split_probability = 0.05  # The probability that a population will split
_chromosome_length = 100  # Length of the chromosome (variables in the one max problem)
_population_size = 200  # The population size
_max_generations = 125  # Algorithm will terminate after this many generations
_crossover_rate = 0.9  # Crossover rate (set between 0.0 and 1.0)
_dom_increase_factor = 0.1  # Top 10% of individuals dominance values increase by this much (set between 0.0 and 1.0)
_dom_decrease_factor = 0.1  # Bottom 10% of individuals dominance values increase by this much (set between 0.0 and 1.0)
_mut_increase_factor = 0.5  # Top 10% of individuals mutation values decrease by this much (set between 0.0 and 1.0)
_mut_decrease_factor = 0.4  # Bottom 10% of individuals mutation values increase by this much (set between 0.0 and 1.0)
number_experiments = 1  # Determines how many experiments we will run in a single execution
experiment_number = 0  # Tracks the number of experiments that have run

# Run the algorithm for
while experiment_number < number_experiments:
    # Create an instance of the numpy random generator for experimental control
    random_gen = RandomGenerator(seed=_seed)

    # Path to where we are storing the results

    # Define the parts of the file path
    results_dir = '../../../results'
    filename = f'mtree_seed_{_seed}_pop_{_population_size}_gen_{_max_generations}_cxp_{_crossover_rate}_domincfac_{round(_dom_increase_factor, 2)}_domdecfac_{round(_dom_decrease_factor, 2)}_mutincfac_{round(_mut_increase_factor, 2)}_mutdecfac_{round(_mut_decrease_factor, 2)}'

    # Construct the full file path
    _results_path = os.path.join(results_dir, filename)

    main(random_gen,
         chromosome_length=_chromosome_length,
         split_probability=_split_probability,
         population_size=_population_size,
         max_generations=_max_generations,
         crossover_rate=_crossover_rate,
         dom_increase_factor=_dom_increase_factor,
         dom_decrease_factor=_dom_decrease_factor,
         mut_increase_factor=_mut_increase_factor,
         mut_decrease_factor=_mut_decrease_factor,
         results_path=_results_path)

    experiment_number += 1  # Increment the experiment counter and track this
    _seed = random_gen.randint(0, 1000)  # Set a new seed for the new experiment
