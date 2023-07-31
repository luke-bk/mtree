import sys

from evolutionary_algorithm.chromosome.Chromosome import Chromosome
from evolutionary_algorithm.genetic_operators.CrossoverOperator import crossover
from evolutionary_algorithm.genetic_operators.MutationOperators import MutationOperators

# Create sample chromosomes for testing
from evolutionary_algorithm.population.Population import Population

parent_name = "Parent"
part_chromosome_length = 4
gene_type = "real"
gene_min = 0.0
gene_max = 1.0

chromosome_one = Chromosome(parent_name, part_chromosome_length, gene_type, gene_min, gene_max)

pop = Population("0", 1)
# Add some chromosomes to the population
# ...
x = 0

while x < 5:
    pop.add_chromosome(Chromosome(parent_name, part_chromosome_length, gene_type, gene_min, gene_max))
    x += 1

# Print the population
pop.print_population()