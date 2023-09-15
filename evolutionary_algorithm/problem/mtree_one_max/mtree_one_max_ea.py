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

    # Create an initial (root node) mtree population (where each individual is a list of integers)
    pop = Population(random_generator=random_generator, name="0", generation=current_generation, fitness=0,
                     parent_population=None)

    # Populate with randomly generated bit chromosomes, of chromosome_length size
    for _ in range(population_size):
        pop.add_chromosome(Chromosome(random_generator, pop.get_name(), chromosome_length, "bit"))

    # Set up the m-ary tree structure
    # Create a root node
    root_region = Region1D(0, chromosome_length - 1)
    binary_tree = BinaryTree(random_generator=random_generator, region=root_region,
                             level=0, parent=None,
                             child_number=0, population=pop,
                             max_depth=3)

    print(f"Start of evolution for seed {random_generator.seed}")

    # Evaluate the entire root population, assign fitness score
    for chromosome in binary_tree.population.chromosomes:
        complete_solution = [chromosome]  # Form complete solution
        chromosome.set_fitness(one_max.fitness_function_mtree(complete_solution))  # Evaluate complete solution
        complete_solution.clear()  # Clear out the complete solution ready for the next evaluation
        total_evaluated += 1  # Increase number of evaluations counter

    print("  ")
    print(f"  Total evaluated {total_evaluated} individuals")

    # Begin the evolutionary loops
    while current_generation < max_generations:
        # Increment generation counter
        current_generation += 1

        print(f"-- Generation {current_generation} --")

        #  For each active population
        for leaf_node in binary_tree.get_leaf([]):
            # Check for split
            # ******TO DO*******

            # Check for merge
            # ******TO DO*******

            # Save best current chromosome
            elite = leaf_node.population.get_chromosome_with_max_fitness()

            # Select the next generation individuals
            new_chromosomes = SelectionOperators.sus_selection_fast_clone(random_generator,
                                                                          leaf_node.population.chromosomes,
                                                                          len(leaf_node.population.chromosomes))

            ParameterManager.modify_dominance_mutation_top_and_bottom_10_percent(random_generator, new_chromosomes,
                                                                                 dom_increase_factor=dom_increase_factor,
                                                                                 dom_decrease_factor=dom_decrease_factor,
                                                                                 mut_increase_factor=mut_increase_factor,
                                                                                 mut_decrease_factor=mut_decrease_factor)

            # Apply crossover to the new chromosomes
            for parent_one, parent_two in zip(new_chromosomes[::2], new_chromosomes[1::2]):
                # cross two individuals with probability crossover_rate
                if random_generator.random() < crossover_rate:
                    CrossoverOperator.crossover(random_generator, parent_one, parent_two)

            # Apply mutation to the new chromosomes
            for mutant in new_chromosomes:
                MutationOperators.perform_bit_flip_mutation(random_generator, mutant)

            # Evaluate the individuals
            for chromosome in new_chromosomes:
                complete_solution = [chromosome]  # Form complete solution
                chromosome.set_fitness(one_max.fitness_function_mtree(complete_solution))  # Evaluate complete solution
                complete_solution.clear()  # Clear out the complete solution ready for the next evaluation
                total_evaluated += 1  # Increase number of evaluations counter

            # Replace old generation with the new generation
            leaf_node.population.chromosomes[:] = new_chromosomes

            # Elitism, add in the elitist individual
            leaf_node.population.chromosomes[-1] = elite

            print(f"  Evaluated {len(leaf_node.population.chromosomes)} individuals")

            fits = [ind.get_fitness() for ind in leaf_node.population.chromosomes]
            # Print the stats (max, min, mean, std) and write out to csv
            results.print_stats_short(leaf_node.population.chromosomes, fits, len(binary_tree.get_leaf([])))
            results.flush()  # Flush the content to the file after each generation

            length = len(leaf_node.population.chromosomes)
            mean = sum(fits) / length
            sum2 = sum(x * x for x in fits)
            std = abs(sum2 / length - mean ** 2) ** 0.5

            print("  Min %s" % min(fits))
            print("  Avg %s" % mean)
            print("  Max %s" % max(fits))
            print("  Std %s" % std)

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
