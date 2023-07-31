import sys

from evolutionary_algorithm.chromosome.Chromosome import Chromosome
from evolutionary_algorithm.genetic_operators.CrossoverOperator import crossover
from evolutionary_algorithm.genetic_operators.MutationOperators import MutationOperators

# Create sample chromosomes for testing
parent_name = "Parent"
part_chromosome_length = 4
gene_type = "real"
gene_min = 0.0
gene_max = 1.0

chromosome_one = Chromosome(parent_name, part_chromosome_length, gene_type, gene_min, gene_max)
chromosome_two = Chromosome(parent_name, part_chromosome_length, gene_type, gene_min, gene_max)


sys.stdout.write("Chromosome one: ")
chromosome_one.print_values()
sys.stdout.write("Chromosome Two: ")
chromosome_two.print_values()

new_chromosome = crossover(chromosome_one, chromosome_two)

# sys.stdout.write("Offspring: ")
new_chromosome.print_values()
