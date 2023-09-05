# Import custom mtree chromosome
from evolutionary_algorithm.chromosome.Chromosome import Chromosome
# Import custom mtree population for splitting/ merging ability
from evolutionary_algorithm.population.Population import Population

# Import custom fitness function
import evolutionary_algorithm.fitness_function.one_max as one_max

# Import custom class for managing experiment results reporting
from evolutionary_algorithm.stats.reporting import ExperimentResults

# Import helper functions related to the evolutionary algorithm
from evolutionary_algorithm.genetic_operators import SelectionOperators, MutationOperators, CrossoverOperator


def hall_of_fame_check(hall_of_fame, pop):
    # After evaluating the individuals, find the best individual in the current population
    best_individual = pop.get_chromosome_with_max_fitness()
    # If the Hall of Fame is empty or the best individual is better than the current best in the Hall of Fame
    if not hall_of_fame or best_individual.get_fitness() > max(hall_of_fame,
                                                               key=lambda x: x.get_fitness()).get_fitness():
        hall_of_fame.clear()  # We only want to keep one at this stage
        hall_of_fame.append(best_individual)  # Add the new best into the hall of fame
    return hall_of_fame[0]


def main(random_generator, chromosome_length, population_size, max_generations, crossover_rate, results_path):
    # Handle reporting (run stats)
    # results = ExperimentResults(random_generator.seed, main_directory=results_path)
    # Initialize the Hall of Fame
    hall_of_fame = []

    # Create an initial (root node) mtree population (where each individual is a list of integers)
    pop = Population("0", 0)

    # Populate with randomly generated bit chromosomes, of chromosome_length size
    for _ in range(population_size):
        pop.add_chromosome(Chromosome(pop.get_name(), chromosome_length, "bit"))

    # Variable keeping track of the number of generations
    current_generation = 0

    print(f"Start of evolution for seed {random_generator.seed}")

    # Evaluate the entire population, assign fitness score
    for individuals in pop.chromosomes:
        individuals.set_fitness(one_max.fitness_function(individuals))

    # Check hall of fame and update
    hall_of_fame[0] = hall_of_fame_check(hall_of_fame, pop)

    print("  ")
    print(f"  Evaluated {len(pop.chromosomes)} individuals")

    # Begin the evolutionary loops
    while current_generation < max_generations:
        # Increment generation counter
        current_generation += 1

        print(f"-- Generation {current_generation} --")

        # Select the next generation individuals
        new_chromosomes = SelectionOperators.sus_selection(pop.chromosomes, len(pop.chromosomes))

        # Apply crossover to the new chromosomes
        for child1, child2 in zip(new_chromosomes[::2], new_chromosomes[1::2]):

            # cross two individuals with probability crossover_rate
            if random_generator.random() < crossover_rate:
                new_chromosomes.append(CrossoverOperator.crossover(child1, child2))
                new_chromosomes.append(CrossoverOperator.crossover(child1, child2))

        # Apply mutation to the new chromsomes
        for mutant in new_chromosomes:
            MutationOperators.perform_bit_flip_mutation(mutant)

        # Evaluate the individuals with an invalid fitness
        for individuals in new_chromosomes:
            individuals.set_fitness(one_max.fitness_function(individuals))

        # Replace old generation with the new generation
        pop.chromosomes = new_chromosomes

        # Check hall of fame and update
        hall_of_fame[0] = hall_of_fame_check(hall_of_fame, pop)

        # Elitism
        pop.chromosomes[-1] = hall_of_fame[0]

        print(f"  Evaluated {len(pop.chromosomes)} individuals")

        fits = [ind.get_fitness() for ind in pop.chromosomes]
        # Print the stats (max, min, mean, std) and write out to csv
        # results.print_stats_short(pop.chromosomes, fits)
        # results.flush()  # Flush the content to the file after each generation

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

    best_ind = pop.get_chromosome_with_max_fitness()
    best_ind.print_values()
    print(f"Best individual fitness: {best_ind.get_fitness()}")

    # Close down reporting
    # results.close()
