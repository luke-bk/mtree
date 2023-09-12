# Handle reporting (run stats)
from evolutionary_algorithm.chromosome.Chromosome import Chromosome
from evolutionary_algorithm.population.Population import Population
from helpers.random_generator import RandomGenerator


# Create an initial (root node) mtree population (where each individual is a list of integers)
from helpers.random_generator import RandomGenerator

random_gen = RandomGenerator(seed=10)

pop = [Population(random_gen, "0", 0, 0)]

# Populate with randomly generated bit chromosomes, of chromosome_length size
for _ in range(6):
    pop[0].add_chromosome(Chromosome(random_gen, pop[0].get_name(), 6, "bit"))

for x in pop[0].chromosomes:
    x.set_fitness(1)

generation = 1  # Specify the generation number
child1, child2 = pop[0].split_population(generation, 0)

# Print the initial population
print("Initial Population:")
pop[0].print_population_simple()

# Print the details of the child populations
print("Child Population 1:")
child1.print_population_simple()

print("\nChild Population 2:")
child2.print_population_simple()