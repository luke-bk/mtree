# Import custom mtree chromosome
from evolutionary_algorithm.chromosome.Chromosome import Chromosome
# Import custom mtree population for splitting/ merging ability
from evolutionary_algorithm.evaluation import Collaboration
from evolutionary_algorithm.evaluation.fitness_function.rosenbrock import rosenbrock
from evolutionary_algorithm.population import Elitism
from evolutionary_algorithm.population.Population import Population

# Import custom fitness function
import evolutionary_algorithm.evaluation.fitness_function.one_max as one_max

# Import custom class for managing experiment results reporting
from evolutionary_algorithm.population.structure.binary_tree.BinaryTree import BinaryTree
from evolutionary_algorithm.population.structure.binary_tree.Region1D import Region1D
from evolutionary_algorithm.stats.reporting import ExperimentResults

# Import helper functions related to the evolutionary algorithm
from evolutionary_algorithm.genetic_operators import SelectionOperators, MutationOperators, CrossoverOperator, \
    ParameterManager


def main(random_generator, dimension, lower_bounds, _upper_bounds, split_probability, merge_threshold, population_size,
         max_generations, crossover_rate, dom_increase_factor, dom_decrease_factor, mut_increase_factor, mut_decrease_factor,
         results_path):
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
                     parent_population=None)  # The root population doesn't have a parent

    # Populate with randomly generated bit chromosomes, of chromosome_length size
    for _ in range(population_size):
        pop.add_chromosome(Chromosome(random_generator, pop.get_name(), dimension, "real", lower_bounds, _upper_bounds))

    # Set up the m-ary tree structure
    # Create a root node
    root_region = Region1D(0, dimension - 1)  # Let's us know which part of the solution its solutions cover
    binary_tree = BinaryTree(random_generator=random_generator,
                             region=root_region,  # Currently evolving solutions for this part of the problem
                             level=0,  # Level in the binary tree structure, 0 for root
                             parent=None,  # Has no parent
                             child_number=0,  # The root node isn't a child, so let's default to 0
                             population=pop,  # The population at this node
                             max_depth=3)  # The max depth the tree can reach

    print(f"Start of evolution for seed {random_generator.seed}")

    # Evaluate the entire root population, assign fitness score
    for chromosome in binary_tree.population.chromosomes:
        complete_solution = [chromosome]  # Form complete solution
        chromosome.set_fitness(rosenbrock(complete_solution))  # Evaluate complete solution
        complete_solution.clear()  # Clear out the complete solution ready for the next evaluation
        total_evaluated += 1  # Increase number of evaluations counter

    # Save best current chromosome
    binary_tree.population.elite = binary_tree.population.get_chromosome_with_min_fitness()
    binary_tree.elite_collaborators = []  # No collaborators as pop = 1
    binary_tree.index_in_collaboration = 0  # Index defaults to 0

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
            binary_tree.select_for_split(current_generation)

        # Check for merge conditions for all active populations
        for leaf_node in binary_tree.get_leaf([]):
            if leaf_node.population.merge_tracker > merge_threshold:  # Merge tracks is greater than the threshold
                binary_tree.select_for_merge(leaf_node.population.get_name())  # Merge the population

        #  For each active population
        for leaf_node in binary_tree.get_leaf([]):
            # Select the next generation individuals
            new_chromosomes = SelectionOperators.sus_selection_fast_clone(random_generator,
                                                                          leaf_node.population.chromosomes,
                                                                          len(leaf_node.population.chromosomes))

            # Selection pressure on the top and bottom 10%. Top 10% chromosomes have their expressed genes mutation
            # rate lowered, and dominance values increased. This is inverse for the bottom 10%.
            ParameterManager.modify_dominance_mutation_top_and_bottom_10_percent(random_generator, new_chromosomes,
                                                                                 dom_increase_factor=dom_increase_factor,
                                                                                 dom_decrease_factor=dom_decrease_factor,
                                                                                 mut_increase_factor=mut_increase_factor,
                                                                                 mut_decrease_factor=mut_decrease_factor)

            # If the population wasn't created this generation
            if leaf_node.population.generation < current_generation:
                # Apply crossover to the new chromosomes
                for parent_one, parent_two in zip(new_chromosomes[::2], new_chromosomes[1::2]):
                    # cross two individuals with probability crossover_rate
                    if random_generator.random() < crossover_rate:
                        CrossoverOperator.crossover(random_generator, parent_one, parent_two)

                # Apply mutation to the new chromosomes
                for mutant in new_chromosomes:
                    MutationOperators.perform_bit_flip_mutation(random_generator, mutant)

            # Get collaborators from each active population except the current one
            complete_solution, sub_solution_index = Collaboration.collaborate(random_generator, binary_tree, leaf_node)

            # Evaluate the individuals
            for chromosome in new_chromosomes:  # Each chromosome in the new pop needs to be evaluated
                temp_collab = complete_solution[:]  # Use slicing to create a copy
                temp_collab.insert(sub_solution_index, chromosome)  # Insert chrom into the solution at index
                chromosome.set_fitness(one_max.fitness_function_mtree(temp_collab))  # Evaluate complete solution
                temp_collab.remove(chromosome)  # Remove the evaluated chromosome from the evaluated list
                total_evaluated += 1  # Increase number of evaluations counter
                total_evaluations_per_generation += 1

            # Replace old generation with the new generation
            leaf_node.population.chromosomes[:] = new_chromosomes

            # Check for new elite
            Elitism.update_elite(leaf_node, complete_solution, sub_solution_index)

            print(f"  Population {leaf_node.population.get_name()}: {len(leaf_node.population.chromosomes)} "
                  f"evaluations")

            fits = [ind.get_fitness() for ind in leaf_node.population.chromosomes]  # Collect all the fitness scores
            total_fitness_per_generation.extend(fits)  # Add to the list

            leaf_node.population.check_if_improved()  # Update the internal population state to see if it has improved

        # Print the stats (max, min, mean, std) and write out to csv
        results.print_stats_short(total_evaluations_per_generation,
                                  total_fitness_per_generation,
                                  len(binary_tree.get_leaf([])),
                                  binary_tree)
        results.flush()  # Flush the content to the file after each generation

    # End of evolutionary process
    print("-- End of (successful) evolution --")

    # After the evolutionary loop generate the fitness plots
    results.plot_fitness_with_target_and_populations(dimension)
    # Print the best solution
    results.find_best_solution(binary_tree)
    # Close down reporting
    results.close()
