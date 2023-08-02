import sys

from evolutionary_algorithm.chromosome.Chromosome import Chromosome
from evolutionary_algorithm.genetic_operators.CrossoverOperator import crossover
from evolutionary_algorithm.genetic_operators.MutationOperators import MutationOperators

# Create sample chromosomes for testing
from evolutionary_algorithm.population.Population import Population
#
#
# def fitness_function(chromosome: Chromosome) -> float:
#     return sum(chromosome.express_highest_dominance())
#
#
# parent_name = "Parent"
# part_chromosome_length = 4
# gene_type = "bit"
# # gene_min = 0.0
# # gene_max = 1.0
#
# # chromosome_one = Chromosome(parent_name, part_chromosome_length, gene_type, gene_min, gene_max)
#
# pop = Population("0", 1)
# # Add some chromosomes to the population
# # ...
# x = 0
#
# while x < 5:
#     # pop.add_chromosome(Chromosome(parent_name, part_chromosome_length, gene_type, gene_min, gene_max))
#     pop.add_chromosome(Chromosome(parent_name, part_chromosome_length, gene_type))
#     x += 1
#
# # Print the population
# pop.print_population()
# pop.print_population_expressed_form()
#
# for inds in pop.chromosomes:
#     inds.set_fitness(fitness_function(inds))
#
# pop.print_population_human_readable()

name = "0"
child_name = name + "0"
print(child_name)
