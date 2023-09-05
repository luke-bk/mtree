# Import various libraries and modules
import numpy as np  # NumPy, primarily used for the chararray

# Import components from the DEAP library for evolutionary algorithms
from deap import base  # Base components for defining individuals and fitness functions
from deap import creator  # Creator module for defining custom types
from deap import tools  # Tools for evolutionary operators and algorithms

# Import custom mtree chromosome
from evolutionary_algorithm.chromosome import Chromosome
# Import custom mtree population for splitting/ merging ability
from evolutionary_algorithm.population import Population

# Import custom fitness function
import evolutionary_algorithm.fitness_function.one_max as one_max

# Import custom class for managing experiment results reporting
from evolutionary_algorithm.stats.reporting import ExperimentResults

# Import helper functions related to the evolutionary algorithm
from evolutionary_algorithm.genetic_operators import SelectionOperators, MutationOperators, CrossoverOperator

# ----------
# Main GA loop
# ----------
from helpers.random_generator import RandomGenerator


def main(random_generator, population_size, max_generations, mutation_rate, crossover_rate, results_path):
    # ----------
    # Create Fitness and Individual classes
    # ----------
    # 1.0 denotes that we are preforming a maximisation task, each individual will have a FitnessMax property
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    # Our individual is a mtree binary chromosome, and they have a FitnessMin property
    creator.create("Individual", Chromosome, fitness=creator.FitnessMax)

    # Access variable to store our various EA properties and operators
    toolbox = base.Toolbox()

    # Register the mtree chromosome constructor as the initialisation.
    # Mtree chromosomes need a parent_name [root for initial], part_chromosome_length, gene_type [bit or real]
    toolbox.register("create_chromosome", Chromosome, parent_name="root", part_chromosome_length=10, gene_type="bit")

    # Structure initializers
    # Register individual initialization, numpy array representations an image
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.create_chromosome)

    # define the population to be a list of individuals, that are numpy arrays representations of images
    toolbox.register("population", tools.initRepeat, Population, toolbox.individual)

    # ----------
    # Operator registration
    # ----------
    # register the objective function (fitness function)
    toolbox.register("evaluate", one_max)

    # register the crossover operator
    toolbox.register("mate", CrossoverOperator.crossover)

    # register a mutation operator
    toolbox.register("mutate", MutationOperators.perform_bit_flip_mutation)

    # register a selection operator
    toolbox.register("select", SelectionOperators.sus_selection)

    # Handle reporting (run stats, and best chromosome)
    results = ExperimentResults(random_generator.seed, main_directory=results_path)

    # create an initial population of 300 individuals (where each individual is a list of integers)
    pop = toolbox.population(n=population_size)

    # Variable keeping track of the number of generations
    current_generation = 0

    print(f"Start of evolution for seed {random_generator.seed}")

    # Evaluate the entire population
    fitness_scores = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitness_scores):
        ind.fitness.values = fit
    print("  ")
    print("  ")
    print(f"  Evaluated {len(pop)} individuals")

    # Begin the evolutionary loops
    while current_generation < max_generations:
        # Increment generation counter
        current_generation += 1

        print(f"-- Generation {current_generation} --")

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability crossover_rate
            if random_generator.random() < crossover_rate:
                toolbox.mate(child1, child2)

                # fitness values of the children must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            # mutate an individual with probability mutation_rate
            if random_generator.random() < mutation_rate:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitness_scores = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitness_scores):
            ind.fitness.values = fit

        print(f"  Evaluated {len(invalid_ind)} individuals")

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitness_scores in one list for stats reasons
        fits = [ind.fitness.values[0] for ind in pop]

        # Print the stats (max, min, mean, std) and write out to csv
        results.print_stats_short(pop, fits)
        results.flush()  # Flush the content to the file after each generation

    # End of evolutionary process
    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]

    print(f"Best individual is {best_ind}, {best_ind, best_ind.fitness.values}")

    # Close down reporting
    results.close()
