from evolutionary_algorithm.chromosome.Chromosome import Chromosome
from evolutionary_algorithm.genetic_operators.MutationOperators import MutationOperators

chromosome = Chromosome("d", 10, 'real', -1.0, 1.0)
clone = chromosome.clone()
#
# print("---Original--------")
# # chromosome.print_values_verbose()
# print(chromosome.express_highest_dominance())
# print("----After Split----")
# split_one, split_two = chromosome.split_chromosome()
#
# # split_one.print_values_verbose()
# # split_two.print_values_verbose()
# print(split_one.express_highest_dominance())
# print(split_two.express_highest_dominance())
#
# print("----After Merge----")
# chromosomeMerged = Chromosome("DaddyTwo", "", 4, 'real', -1.0, 1.0)
# chromosomeMerged.merge_chromosome(split_one, split_two)
#
# print(chromosomeMerged.express_highest_dominance())

# print("--------Original--------")
# print(chromosome.express_highest_dominance())
# print("--------Clone--------")
# print(clone.express_highest_dominance())
# print("--------Mutate--------")
# # Perform Gaussian mutation
# MutationOperators.perform_bit_flip_mutation(chromosome)
# # MutationOperators.perform_gaussian_mutation(chromosome, 0.0, 0.1)
# print("--------Mutated--------")
# print(chromosome.express_highest_dominance())
# print("--------Clone--------")
# print(clone.express_highest_dominance())

x = 0

while x < 20:
    MutationOperators.perform_gaussian_mutation(chromosome, 0.0, 0.1)
    x += 1
