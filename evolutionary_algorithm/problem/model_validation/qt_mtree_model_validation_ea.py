# Import custom mtree chromosome
import cv2

from evolutionary_algorithm.chromosome.ChromosomeReal import ChromosomeReal
# Import custom mtree population for splitting/ merging ability
from evolutionary_algorithm.evaluation import Collaboration
from evolutionary_algorithm.population import Elitism
from evolutionary_algorithm.population.Population import Population

# Import custom fitness function
from evolutionary_algorithm.evaluation.fitness_function.manhattan_distance import manhattan_distance_fitness

# Import custom class for managing experiment results reporting
from evolutionary_algorithm.population.structure.quad_tree.QuadTree import QuadTree
from evolutionary_algorithm.population.structure.quad_tree.Region import Region
from evolutionary_algorithm.stats.reporting import ExperimentResults

# Import helper functions related to the evolutionary algorithm
from evolutionary_algorithm.genetic_operators import SelectionOperators, MutationOperators, CrossoverOperator, \
    ParameterManager


def main(loaded_model, random_generator, is_minimization_task, split_probability, merge_threshold, population_size,
         max_generations, crossover_rate, mutation_rate, results_path, base_image, current_class):
    # Handle reporting (run stats)
    results = ExperimentResults(random_generator.seed, main_directory=results_path)

    # Variable keeping track of the number of generations
    current_generation = 0
    # Variable keeping track of the number evaluations
    total_evaluated = 0
    # Variable keeping track of total fitness per generation
    total_fitness_per_generation = []

    # Create an initial mtree population (where each individual is a list of bits)
    pop = Population(random_generator=random_generator,  # Single random generator for the whole experiment
                     name="0",  # Root population should always be "0"
                     generation=current_generation,  # Track when the population was created
                     fitness=0,  # Track what is the current best fitness score
                     parent_population=None,  # The root population doesn't have a parent
                     is_minimization_task=is_minimization_task)  # Min or max problem?

    # Populate with randomly generated bit chromosomes, of chromosome_length size
    # for _ in range(population_size):
    #     pop.add_chromosome(ChromosomeReal(random_generator, pop.get_name(), "../../../images/test_images/base_9.png"))
    while len(pop.chromosomes) < population_size:
        # Create a new chromosome
        new_chromosome = ChromosomeReal(random_generator, pop.get_name(), base_image)

        # Load the comparison image
        comparison_image = cv2.imread(base_image, cv2.IMREAD_GRAYSCALE)

        # Calculate the Manhattan distance
        score = manhattan_distance_fitness(loaded_model, new_chromosome.chromosome, comparison_image, current_class)

        # Add the chromosome to the population if the score isn't 999,999
        if score != 999999:
            pop.add_chromosome(new_chromosome)

    # Set up the m-ary tree structure
    # Create a root node
    root_region = Region(0, 0, pop.chromosomes[0].length - 1,
                         pop.chromosomes[0].length - 1)  # Lets us know which part of the solution its solutions cover
    quad_tree = QuadTree(random_generator=random_generator,
                         region=root_region,  # Currently evolving solutions for this part of the problem
                         level=0,  # Level in the binary tree structure, 0 for root
                         parent=None,  # Has no parent
                         child_number=0,  # The root node isn't a child, so let's default to 0
                         population=pop,  # The population at this node
                         max_depth=3)  # The max depth the tree can reach

    print(f"Start of evolution for seed {random_generator.seed}")

    # Evaluate the entire root population, assign fitness score
    for chromosome in quad_tree.population.chromosomes:
        complete_solution = [chromosome]  # Form complete solution
        chromosome.set_fitness(manhattan_distance_fitness(loaded_model, chromosome.chromosome,
                                                          cv2.imread(base_image,
                                                                     cv2.IMREAD_GRAYSCALE),
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
        # if random_generator.uniform(0.0, 1.0) < split_probability:
        if current_generation == 5:
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
                    MutationOperators.perform_gaussian_mutation_image(random_generator,
                                                                      mutant.chromosome,
                                                                      mutation_rate,
                                                                      0.00,
                                                                      0.1)

            # Get collaborators from each active population except the current one
            complete_solution, sub_solution_index = Collaboration.collaborate(random_generator,
                                                                                      quad_tree,
                                                                                      leaf_node)

            # if current_generation == 7:
            #     print("Collaboration: ", sub_solution_index, " ", complete_solution)
            #     for c in complete_solution:
            #         print(c.parent_name)

            # Evaluate the individuals
            for chromosome in new_chromosomes:  # Each chromosome in the new pop needs to be evaluated
                # Evaluate complete solution
                temp_collab = complete_solution[:]  # Use slicing to create a copy
                temp_collab.insert(sub_solution_index, chromosome)  # Insert chrom into the solution at index

                chromosome.set_fitness(manhattan_distance_fitness(
                    # send in the model
                    loaded_model,
                    # Form collaboration
                    Collaboration.collaborate_image(temp_collab),
                    # The base image we are changing classes
                    cv2.imread(base_image,
                               # Read it in as greyscale
                               cv2.IMREAD_GRAYSCALE),
                    current_class))

                temp_collab.remove(chromosome)  # Remove the evaluated chromosome from the evaluated list

                #
                # if len(quad_tree.get_leaf([])) > 1:
                #     chromosome.set_fitness(manhattan_distance_fitness(  # send in the model
                #         loaded_model,
                #         # Form collaboration
                #         Collaboration.collaborate_image(random_generator,
                #                                         quad_tree),
                #         # The base image we are changing classes
                #         cv2.imread("../../../images/test_images/base_9.png",
                #                    # Read it in as greyscale
                #                    cv2.IMREAD_GRAYSCALE)))
                # else:
                #     chromosome.set_fitness(manhattan_distance_fitness(  # send in the model
                #         loaded_model,
                #         # Form collaboration
                #         chromosome.chromosome,
                #         # The base image we are changing classes
                #         cv2.imread("../../../images/test_images/base_9.png",
                #                    # Read it in as greyscale
                #                    cv2.IMREAD_GRAYSCALE)))

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
    results.plot_fitness_with_target_and_populations_min_task_zoom(0)  # Print the best solution
    # Print the best solution
    results.find_best_solution_image(quad_tree)

    # Save the modified image
    output_path = results_path + "_evolved.jpg"
    cv2.imwrite(output_path, quad_tree.population.elite.chromosome)

    # Close down reporting
    results.close()
