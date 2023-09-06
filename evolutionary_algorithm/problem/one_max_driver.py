from evolutionary_algorithm.problem.one_max_ea import main
from helpers.random_generator import RandomGenerator

import cProfile

_seed = 1  # Set the seed for experiment repeatability
_chromosome_length = 600  # Length of the chromosome (variables in the one max problem)
_population_size = 100  # The population size
_max_generations = 100  # Algorithm will terminate after this many generations
_crossover_rate = 0.9  # Crossover rate (set between 0.0 and 1.0)
number_experiments = 1  # Determines how much experiments we will run in a single execution
experiment_number = 0  # Tracks the number of experiments that have run

# Path to where we are storing the results
_results_path = '../../results/seed_' + str(_seed) + "_pop_" + str(_population_size) + "_gen_" + str(_max_generations) \
                + "_cxp_" + str(_crossover_rate)

# Run the algorithm for
while experiment_number < number_experiments:
    # Create a instance of the numpy random generator for experimental control
    random_gen = RandomGenerator(seed=_seed)

    # cProfile.run(
    #     "main(random_gen, chromosome_length=_chromosome_length, population_size=_population_size, max_generations=_max_generations, crossover_rate=_crossover_rate, results_path=_results_path)")
    main(random_gen,
         chromosome_length=_chromosome_length,
         population_size=_population_size,
         max_generations=_max_generations,
         crossover_rate=_crossover_rate,
         results_path=_results_path)

    experiment_number += 1  # Increment the experiment counter and track this
    _seed = random_gen.randint(0, 1000)  # Set a new seed for the new experiment
