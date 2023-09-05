# Import various libraries and modules
import random
import cv2  # OpenCV's library for computer vision stuff
import numpy as np  # NumPy, primarily used for the chararray

# Import components from the DEAP library for evolutionary algorithms
from deap import base  # Base components for defining individuals and fitness functions
from deap import creator  # Creator module for defining custom types
from deap import tools  # Tools for evolutionary operators and algorithms

# Import custom fitness function
import src.code.helpers.temp_helpers
from src.code.evolution.fitness_function.manhattan_distance import manhattan_distance_fitness

# Import helper functions for the evolutionary process
from src.code.evolution.genetic_operators import cx_two_point, mut_uniform_int, mutate_with_noise, \
    mutate_with_noise_fast

# Import custom class for managing experiment results reporting
from src.code.evolution.reporting import ExperimentResults

# Import helper functions related to generating noise and visualisation of the results
from src.code.helpers.image_manipulation import add_clustered_noise_same_colour_profile_to_array, \
    visual_difference_4x4_plot, image_save_interval, \
    add_clustered_noise_random_ellipse_to_array, add_clustered_noise_random_ellipse_fast_to_array, \
    add_clustered_noise_random_ellipse_to_array_mut, add_clustered_noise_random_ellipse_fast_to_array_mut

# ----------
# Main GA loop
# ----------
from src.code.helpers.random_generator import RandomGenerator


def main(random_generator, population_size, max_generations, mutation_rate, crossover_rate, target_image_path,
         noise_base_image_path, output_image_path, results_path, images_to_save):
    # Selection still uses random, want to change in future
    random.seed(random_generator.seed)

    # ----------
    # Create Fitness and Individual classes
    # ----------
    # -1.0 denotes that we are preforming a minimisation task, each individual will have a FitnessMin property
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    # Our individual is a np.array of numbers shape x, y z, and they have a FitnessMin property
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

    # Access variable to store our various EA properties and operators
    toolbox = base.Toolbox()

    # Register the noise generation method as the gene initialization, this takes in a path to the target image
    # and will generate a series of chromosomes by adding noise to this image
    # toolbox.register("gene", add_clustered_noise_same_colour_profile_to_array,
    #                  image_to_change_path=noise_base_image_path, random_generator=random_generator,
    #                  num_of_clusters=3)
    # toolbox.register("gene", add_clustered_noise_random_ellipse_to_array, random_generator=random_generator,
    #                  image_to_change_path=noise_base_image_path, num_clusters=1, min_cluster_density=2,
    #                  max_cluster_density=10, min_intensity=-10, max_intensity=20, min_pix_neighborhood=-1,
    #                  max_pix_neighborhood=1, min_ellipse_length=10, max_ellipse_length=30, min_ellipse_width=5,
    #                  max_ellipse_width=30)
    toolbox.register("gene", add_clustered_noise_random_ellipse_fast_to_array, random_generator=random_generator,
                     image_to_change_path=noise_base_image_path, min_intensity=-200, max_intensity=220,
                     min_ellipse_length=10, max_ellipse_length=30, min_ellipse_width=5, max_ellipse_width=30,
                     min_changes=100, max_changes=255)

    # Structure initializers
    # Register individual initialization, numpy array representations an image
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.gene)

    # define the population to be a list of individuals, that are numpy arrays representations of images
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # ----------
    # Operator registration
    # ----------
    # register the objective function (fitness function)
    target_image = cv2.imread(target_image_path)
    toolbox.register("evaluate", manhattan_distance_fitness, image_two=target_image)

    # register the crossover operator
    toolbox.register("mate", cx_two_point, random_generator=random_generator)

    # register a mutation operator
    # toolbox.register("mutate", mut_uniform_int, low=0, up=255, indpb=0.9, random_generator=random_generator)
    # toolbox.register("mutate", mutate_with_noise, num_clusters=10,
    #                  min_cluster_density=40, max_cluster_density=80, min_intensity=-5, max_intensity=5,
    #                  min_pix_neighborhood=-5, max_pix_neighborhood=10, min_ellipse_length=5, max_ellipse_length=10,
    #                  min_ellipse_width=5, max_ellipse_width=10, indpb=0.01)

    toolbox.register("mutate", mutate_with_noise_fast, random_generator=random_generator,
                     min_intensity=-200, max_intensity=220,
                     min_ellipse_length=10, max_ellipse_length=30, min_ellipse_width=5, max_ellipse_width=30,
                     min_changes=50, max_changes=80)

    # register a selection operator
    toolbox.register("select", tools.selTournament, tournsize=2)

    # Handle reporting (run stats, and best chromosome)
    results = ExperimentResults(random_generator.seed, main_directory=results_path)

    # create an initial population of 300 individuals (where each individual is a list of integers)
    pop = toolbox.population(n=population_size)

    # Variable keeping track of the number of generations
    current_generation = 0

    # Track which gen the images are saved out for the final presentation of results
    images_saved_by_gen = []

    print("Start of evolution for seed %i" % random_generator.seed)

    # Evaluate the entire population
    fitness_scores = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitness_scores):
        ind.fitness.values = fit
    print("  ")
    print("  ")
    print("  Evaluated %i individuals" % len(pop))

    # Begin the evolutionary loops
    while current_generation < max_generations:
        # Increment generation counter
        current_generation += 1

        print("-- Generation %i --" % current_generation)

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

        print("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitness_scores in one list for stats reasons
        fits = [ind.fitness.values[0] for ind in pop]

        # Print the stats (max, min, mean, std) and write out to csv
        results.print_stats_short(pop, fits)
        results.flush()  # Flush the content to the file after each generation

        # save best image for the set interval
        if current_generation % image_save_interval(max_generations, images_to_save) == 0 or current_generation == 1:
            best_ind = tools.selBest(pop, 1)[0]
            # Save the modified image
            output_path = output_image_path + "_gen_" + str(current_generation) + "_evolved.jpg"
            cv2.imwrite(output_path, best_ind)
            images_saved_by_gen.append(current_generation)

    # End of evolutionary process
    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]

    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

    # Save the modified image
    output_path = output_image_path + "_evolved.jpg"
    cv2.imwrite(output_path, best_ind)

    # Display a 4x4 plot of the evolved images and differences
    visual_difference_4x4_plot(target_image_path,
                               output_path,
                               images_saved_by_gen,
                               output_image_path)

    # Close down reporting
    results.close()
