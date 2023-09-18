# Import custom mtree chromosome
from evolutionary_algorithm.chromosome.Chromosome import Chromosome
# Import custom mtree population for splitting/ merging ability
from evolutionary_algorithm.population.Population import Population

# Import custom fitness function
import evolutionary_algorithm.fitness_function.one_max as one_max

# Import custom class for managing experiment results reporting
from evolutionary_algorithm.population.structure.binary_tree.BinaryTree import BinaryTree
from evolutionary_algorithm.population.structure.binary_tree.Region1D import Region1D
from evolutionary_algorithm.stats.reporting import ExperimentResults

# Import helper functions related to the evolutionary algorithm
from evolutionary_algorithm.genetic_operators import SelectionOperators, MutationOperators, CrossoverOperator, \
    ParameterManager


def main(random_generator, chromosome_length, population_size, max_generations, crossover_rate, dom_increase_factor,
         dom_decrease_factor, mut_increase_factor, mut_decrease_factor, results_path):
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
        pop.add_chromosome(Chromosome(random_generator, pop.get_name(), chromosome_length, "bit"))

    # Set up the m-ary tree structure
    # Create a root node
    root_region = Region1D(0, chromosome_length - 1)  # Let's us know which part of the solution its solutions cover
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
        chromosome.set_fitness(one_max.fitness_function_mtree(complete_solution))  # Evaluate complete solution
        complete_solution.clear()  # Clear out the complete solution ready for the next evaluation
        total_evaluated += 1  # Increase number of evaluations counter

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
        if current_generation == 20:
            binary_tree.select_for_split(current_generation)

        # Check for merge
        # if current_generation == 60:
        #     binary_tree.select_for_merge("00")
        #     binary_tree.select_for_merge("01")

        #  For each active population
        for leaf_node in binary_tree.get_leaf([]):
            # Save best current chromosome
            leaf_node.population.elite = leaf_node.population.get_chromosome_with_max_fitness()

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
            complete_solution = []  # Empty complete solution
            sub_solution_index = None  # We need to save our evaluated solution index, so we know where to insert
            for index, collaborator_node in enumerate(binary_tree.get_leaf([])):
                if collaborator_node.population is not leaf_node.population:  # Check if it's not the current population
                    if collaborator_node.population.elite is not None:  # If the elite exists
                        # Add elites from other populations
                        complete_solution.append(collaborator_node.population.elite)
                    else:  # Return a random choice
                        complete_solution.append(random_generator.choice(collaborator_node.population.chromosomes))
                else:
                    sub_solution_index = index  # Save the index where the solution needs to be inserted

            # Evaluate the individuals
            for chromosome in new_chromosomes:  # Each chromosome in the new pop needs to be evaluated
                complete_solution.insert(sub_solution_index, chromosome)  # Insert chrom into the solution at index
                chromosome.set_fitness(one_max.fitness_function_mtree(complete_solution))  # Evaluate complete solution
                complete_solution.remove(chromosome)  # Remove the evaluated chromosome from the evaluated list
                total_evaluated += 1  # Increase number of evaluations counter
                total_evaluations_per_generation += 1

            # Replace old generation with the new generation
            leaf_node.population.chromosomes[:] = new_chromosomes

            # Check for new elite
            current_best = leaf_node.population.get_chromosome_with_max_fitness()
            if leaf_node.population.elite.get_fitness() <= current_best.get_fitness():
                leaf_node.population.elite = current_best

            # Elitism, add in the elitist individual
            leaf_node.population.chromosomes[-1] = leaf_node.population.elite

            print(f"  Evaluated {len(leaf_node.population.chromosomes)} individuals")

            fits = [ind.get_fitness() for ind in leaf_node.population.chromosomes]  # Colelct all the fitness scores
            total_fitness_per_generation.extend(fits)  # Add to the list

        # Print the stats (max, min, mean, std) and write out to csv
        results.print_stats_short(total_evaluations_per_generation, total_fitness_per_generation, len(binary_tree.get_leaf([])))
        results.flush()  # Flush the content to the file after each generation

    # End of evolutionary process
    print("-- End of (successful) evolution --")

    # After the evolutionary loop generate the fitness plots
    # results.plot_fitness_with_target(chromosome_length)
    results.plot_fitness_with_target_and_populations(chromosome_length)

    # best_ind = pop[0].get_chromosome_with_max_fitness()
    # best_ind.print_values_expressed()
    # print(f"Best individual fitness: {best_ind.get_fitness()}")

    # Close down reporting
    results.close()
