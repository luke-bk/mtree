# Import custom mtree chromosome
from evolutionary_algorithm.chromosome.Chromosome import Chromosome
# Import custom mtree population for splitting/ merging ability
from evolutionary_algorithm.population.Population import Population

# Import custom fitness function
import evolutionary_algorithm.fitness_function.one_max as one_max

# Import custom class for managing experiment results reporting
from evolutionary_algorithm.stats.reporting import ExperimentResults

# Import helper functions related to the evolutionary algorithm
from evolutionary_algorithm.genetic_operators import SelectionOperators, MutationOperators, CrossoverOperator, ParameterManager


def main(random_generator, chromosome_length, population_size, max_generations, crossover_rate, dom_increase_factor,
         dom_decrease_factor, mut_increase_factor, mut_decrease_factor, results_path):
    # Handle reporting (run stats)
    results = ExperimentResults(random_generator.seed, main_directory=results_path)

    # Create an initial (root node) mtree population (where each individual is a list of integers)
    pop = Population("0", 0)

    # Populate with randomly generated bit chromosomes, of chromosome_length size
    for _ in range(population_size):
        pop.add_chromosome(Chromosome(random_generator, pop.get_name(), chromosome_length, "bit"))

    # Variable keeping track of the number of generations
    current_generation = 0

    print(f"Start of evolution for seed {random_generator.seed}")

    # Evaluate the entire population, assign fitness score
    for individuals in pop.chromosomes:
        individuals.set_fitness(one_max.fitness_function(individuals))

    print("  ")
    print(f"  Evaluated {len(pop.chromosomes)} individuals")

    # Begin the evolutionary loops
    while current_generation < max_generations:
        # Increment generation counter
        current_generation += 1

        # Save best current chromosome
        elite = pop.get_chromosome_with_max_fitness()

        print(f"-- Generation {current_generation} --")
        # Select the next generation individuals
        new_chromosomes = SelectionOperators.sus_selection_fast_clone(random_generator,
                                                                      pop.chromosomes,
                                                                      len(pop.chromosomes))

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
        for individuals in new_chromosomes:
            individuals.set_fitness(one_max.fitness_function(individuals))

        # Replace old generation with the new generation
        pop.chromosomes[:] = new_chromosomes

        # Elitism, add in the elitist individual
        pop.chromosomes[-1] = elite

        print(f"  Evaluated {len(pop.chromosomes)} individuals")

        fits = [ind.get_fitness() for ind in pop.chromosomes]
        # Print the stats (max, min, mean, std) and write out to csv
        results.print_stats_short(pop.chromosomes, fits)
        results.flush()  # Flush the content to the file after each generation

        length = len(pop.chromosomes)
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
    results.plot_fitness_with_target(chromosome_length)

    best_ind = pop.get_chromosome_with_max_fitness()
    best_ind.print_values_expressed()
    print(f"Best individual fitness: {best_ind.get_fitness()}")

    # Close down reporting
    results.close()
