# Import custom mtree chromosome
import os

# For reading, compressing and saving out DICOM images
import pydicom
from pydicom.uid import RLELossless

from evolutionary_algorithm.chromosome.ChromosomeReal import ChromosomeReal
# Import custom mtree population for splitting/ merging ability
from evolutionary_algorithm.evaluation import Collaboration
from evolutionary_algorithm.population import Elitism
from evolutionary_algorithm.population.Population import Population

# Import custom fitness function
from evolutionary_algorithm.evaluation.fitness_function.manhattan_distance import manhattan_distance_fitness, \
    manhattan_distance_fitness_dcm

# Import custom class for managing experiment results reporting
from evolutionary_algorithm.population.structure.quad_tree.QuadTree import QuadTree
from evolutionary_algorithm.population.structure.quad_tree.Region import Region
from evolutionary_algorithm.stats.reporting import ExperimentResults

# Import helper functions related to the evolutionary algorithm
from evolutionary_algorithm.genetic_operators import SelectionOperators, MutationOperators, CrossoverOperator, \
    ParameterManager


def main(loaded_model, random_generator, is_minimization_task, split_probability, merge_threshold, population_size,
         max_generations, crossover_rate, mutation_rate, results_path, base_image, image_type, current_class):
    # Handle reporting (run stats)
    results = ExperimentResults(random_generator.seed, main_directory=results_path)

    # Variable keeping track of the number of generations
    current_generation = 0
    # Variable keeping track of the number evaluations
    total_evaluated = 0
    # Variable keeping track of total fitness per generation
    total_fitness_per_generation = []
    # Current best fitness
    current_best_fitness = 999999999999

    # Create an initial mtree population (where each individual is 'm x n' of an image)
    pop = Population(random_generator=random_generator,  # Single random generator for the whole experiment
                     name="0",  # Root population should always be "0"
                     generation=current_generation,  # Track when the population was created
                     fitness=0,  # Track what is the current best fitness score
                     parent_population=None,  # The root population doesn't have a parent
                     is_minimization_task=is_minimization_task)  # Min or max problem?

    # Populate with randomly generated 'n x m' chromosomes, of population_size
    while len(pop.chromosomes) < population_size:
        # Create a new chromosome from the base image
        new_chromosome = ChromosomeReal(random_generator, pop.get_name(), base_image, image_type)

        # Read the comparison DICOM image
        dicom_data = pydicom.dcmread(base_image)
        comparison_image = dicom_data.pixel_array

        # Calculate the Manhattan distance
        score = manhattan_distance_fitness_dcm(loaded_model,
                                               new_chromosome.chromosome,
                                               comparison_image,
                                               current_class)

        # Add the chromosome to the population if the score isn't 999,999,999
        while score == 999999999:
            # Continue to add noise, until the model thinks it is a different class from its base image
            MutationOperators.perform_gaussian_mutation_dcm_patch(random_generator,
                                                                  new_chromosome.chromosome,
                                                                  0.5,
                                                                  0.00,
                                                                  90.1)

            # Calculate the Manhattan distance
            score = manhattan_distance_fitness_dcm(loaded_model,  # The DFO model
                                                   new_chromosome.chromosome,  # Candidate solution
                                                   comparison_image,  # Base image in array representation
                                                   current_class)  # The class of the base image, we want to flip this
        pop.add_chromosome(new_chromosome)  # Once the chromosome flips the class, we add it to population

    # Set up the solution region and m-ary tree structure
    # Lets us know which part of the solution its solutions cover x1 = 0, x2 = dimension of image etc
    root_region = Region(0, 0, pop.chromosomes[0].length - 1, pop.chromosomes[0].length - 1)
    # Create a root node
    quad_tree = QuadTree(random_generator=random_generator,  # Same random generator for consistency
                         region=root_region,  # Currently evolving solutions for this part of the problem
                         level=0,  # Level in the binary tree structure, 0 for root
                         parent=None,  # Has no parent
                         child_number=0,  # The root node isn't a child, so let's default to 0
                         population=pop,  # The population at this node
                         max_depth=2)  # The max depth the tree can reach, 2 translates to a max of 16 quads

    print(f"Start of evolution for seed {random_generator.seed}")

    # Convert base DICOM image to an array of pixels to compare using the fitness function (manhattan_distance)
    dicom_data = pydicom.dcmread(base_image)
    comparison_image = dicom_data.pixel_array

    # Evaluate the entire root population, assign fitness score
    for chromosome in quad_tree.population.chromosomes:
        complete_solution = [chromosome]  # Form complete solution

        chromosome.set_fitness(manhattan_distance_fitness_dcm(loaded_model,
                                                              chromosome.chromosome,
                                                              comparison_image,
                                                              current_class))  # Evaluate complete solution
        complete_solution.clear()  # Clear out the complete solution ready for the next evaluation
        total_evaluated += 1  # Increase number of evaluations counter

    # Save best current chromosome
    quad_tree.population.elite = quad_tree.population.get_chromosome_with_min_fitness()
    quad_tree.elite_collaborators = []  # No collaborators as pop = 1
    quad_tree.index_in_collaboration = 0  # Index defaults to 0

    print("  ")
    print(f"  Total evaluated {total_evaluated} individuals")

    # Begin the evolutionary loop and run until a max generations limit has been reached, then terminate
    while current_generation < max_generations:
        # Increment generation counter
        current_generation += 1
        # Clear previous generations' fitness scores and evaluation counter
        total_fitness_per_generation.clear()
        # Variable keeping track of total evaluations per generation
        total_evaluations_per_generation = 0

        print(f"-- Generation {current_generation} --")

        # Check for split
        if random_generator.uniform(0.0, 1.0) < split_probability:
            # if current_generation == 5:
            quad_tree.select_for_split(current_generation)

        # Check for merge conditions for all active populations
        for leaf_node in quad_tree.get_leaf([]):
            if leaf_node.population.merge_tracker > merge_threshold:  # Merge tracks is greater than the threshold
                quad_tree.select_for_merge(leaf_node.population.get_name())  # Merge the population

        #  For each active population
        for leaf_node in quad_tree.get_leaf([]):
            # Select the next generation individuals
            new_chromosomes = SelectionOperators.tournament_selection(random_generator,
                                                                      leaf_node.population.chromosomes,
                                                                      2,
                                                                      len(leaf_node.population.chromosomes),
                                                                      is_minimization_task)

            # If the population wasn't created this generation
            if leaf_node.population.generation < current_generation:
                # Apply crossover to the new chromosomes
                for parent_one, parent_two in zip(new_chromosomes[::2], new_chromosomes[1::2]):
                    # cross two individuals with probability crossover_rate
                    if random_generator.random() < crossover_rate:
                        CrossoverOperator.crossover_image(random_generator, parent_one, parent_two)

                # Apply mutation to the new chromosomes
                for mutant in new_chromosomes:
                    if (current_generation < max_generations // 2):
                        MutationOperators.perform_gaussian_mutation_dcm_patch(random_generator,
                                                                              mutant.chromosome,
                                                                              mutation_rate,
                                                                              0.00,
                                                                              30.1)
                    else:
                        MutationOperators.replace_patch_from_original_quad_safe(random_generator,
                                                                                comparison_image,
                                                                                mutant.chromosome,
                                                                                # quad x
                                                                                leaf_node.region.x1,
                                                                                # quad y
                                                                                leaf_node.region.y1,
                                                                                # x length
                                                                                (leaf_node.region.x2 - leaf_node.region.x1) // 2,
                                                                                # y length
                                                                                (leaf_node.region.y2 - leaf_node.region.y1) // 2
                                                                                )

            # Get collaborators from each active population except the current one
            complete_solution, sub_solution_index = Collaboration.collaborate(random_generator,
                                                                              quad_tree,
                                                                              leaf_node)

            # Evaluate the individuals
            for chromosome in new_chromosomes:  # Each chromosome in the new pop needs to be evaluated
                # Evaluate complete solution
                temp_collab = complete_solution[:]  # Use slicing to create a copy
                temp_collab.insert(sub_solution_index, chromosome)  # Insert chrom into the solution at index

                chromosome.set_fitness(manhattan_distance_fitness_dcm(
                    # send in the model
                    loaded_model,
                    # Form collaboration
                    Collaboration.collaborate_image_new(temp_collab, current_generation),
                    # The base image we are changing classes
                    comparison_image,
                    current_class))

                # Check for best fitness
                if chromosome.get_fitness() < current_best_fitness:
                    current_best_fitness = chromosome.get_fitness()

                    ds = pydicom.dcmread(base_image)
                    ds.PixelData = Collaboration.collaborate_image_new(temp_collab, current_generation).astype('uint16')
                    ds.compress(RLELossless, Collaboration.collaborate_image_new(temp_collab, current_generation))
                    file = str(current_generation) + '_' + str(
                        int(current_best_fitness)) + '_best_chromosome_evolved.dcm'
                    dicom_file = os.path.join(results_path, file)
                    ds.save_as(dicom_file)

                temp_collab.remove(chromosome)  # Remove the evaluated chromosome from the evaluated list

                total_evaluated += 1  # Increase number of evaluations counter
                total_evaluations_per_generation += 1

            # Replace old generation with the new generation
            leaf_node.population.chromosomes[:] = new_chromosomes

            # Check for new elite
            Elitism.update_elite_min_image(leaf_node, complete_solution)

            print(f"  Population {leaf_node.population.get_name()}: {len(leaf_node.population.chromosomes)} "
                  f"evaluations")

            fits = [ind.get_fitness() for ind in leaf_node.population.chromosomes]  # Collect all the fitness scores
            total_fitness_per_generation.extend(fits)  # Add to the list

            leaf_node.population.check_if_improved()  # Update the internal population state to see if it has improved

        # Print the stats (max, min, mean, std) and write out to csv
        results.print_stats_short(total_evaluations_per_generation,
                                  total_fitness_per_generation,
                                  len(quad_tree.get_leaf([])),
                                  quad_tree)
        results.flush()  # Flush the content to the file after each generation

    # End of evolutionary process
    print("-- End of (successful) evolution --")

    # After the evolutionary loop generate the fitness plots
    results.plot_fitness_with_target_and_populations_min_task(0)
    results.plot_fitness_with_target_and_populations_min_task_zoom(0)
    results.plot_fitness_with_target_and_populations_min_isolated(0)
    # Print the best solution
    results.find_best_solution_image(quad_tree)
    # results.find_best_solution_image_dcm(quad_tree, base_image)

    # Close down reporting
    results.close()
