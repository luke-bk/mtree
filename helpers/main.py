from evolutionary_algorithm.chromosome.Chromosome import Chromosome

chromosome = Chromosome("Daddy", "", 3, 'real', -1.0, 1.0)
clone = chromosome.clone()

print("---Original--------")
# chromosome.print_values_verbose()
print(chromosome.express_highest_dominance())
print("----After Split----")
split_one, split_two = chromosome.split_chromosome()

# split_one.print_values_verbose()
# split_two.print_values_verbose()
print(split_one.express_highest_dominance())
print(split_two.express_highest_dominance())

print("----After Merge----")
chromosomeMerged = Chromosome("DaddyTwo", "", 4, 'real', -1.0, 1.0)
chromosomeMerged.merge_chromosome(split_one, split_two)

print(chromosomeMerged.express_highest_dominance())