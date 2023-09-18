from evolutionary_algorithm.population.structure.binary_tree.BinaryTree import BinaryTree
from evolutionary_algorithm.population.structure.binary_tree.Region1D import Region1D
from helpers.random_generator import RandomGenerator
from evolutionary_algorithm.population.Population import Population
from evolutionary_algorithm.chromosome.Chromosome import Chromosome

# Create a random generator
random_generator = RandomGenerator(seed=1)

gen = 0
# Create a root node
root_region = Region1D(0, 23)
root_pop = Population(random_generator=random_generator, name="0", generation=gen, fitness=0, parent_population=None)
for _ in range(4):
    root_pop.add_chromosome(Chromosome(random_generator, root_pop.get_name(), 8, "real", gene_min=-1.0, gene_max=1.0))

for x in root_pop.chromosomes:
    x.set_fitness(1)

binary_tree = BinaryTree(random_generator=random_generator, region=root_region,
                         level=0, parent=None,
                         child_number=0, population=root_pop,
                         max_depth=3)

# print("---------------------------------TREE CREATED---------------------------")
# print(binary_tree.print_tree())
# for x in binary_tree.get_leaf([]):
#     print(x.print_self())
#     x.population.print_population_simple()
#
# print(binary_tree.get_leaf([])[0].population.chromosomes[0].part_chromosomes[0].genes[0].gene_value)

print("---------------------------------SPLIT TREE---------------------------")
binary_tree.select_for_split(0)
# print(binary_tree.print_tree())
for x in binary_tree.get_leaf([]):
    x.population.elite = x.population.chromosomes[0]
    print(x.print_self())
    # x.population.print_population_simple()

# # Change chromosome value
#
# print(binary_tree.get_leaf([])[0].population.chromosomes[0].part_chromosomes[0].genes[0].gene_value)
# binary_tree.get_leaf([])[0].population.chromosomes[0].part_chromosomes[0].genes[0].gene_value = 0.99
#
# print(binary_tree.print_tree())
# for x in binary_tree.get_leaf([]):
#     print(x.print_self())
#     x.population.print_population_simple()
#
# print("---------------------------------SPLIT TREE---------------------------")
# binary_tree.select_for_split(0)
#
# print(binary_tree.print_tree())
# for x in binary_tree.get_leaf([]):
#     print(x.print_self())
#     x.population.print_population_simple()
#
# print("---------------------------------MERGE TREE---------------------------")
# binary_tree.select_for_merge("010")
# binary_tree.select_for_merge("011")
#
# print(binary_tree.print_tree())
# for x in binary_tree.get_leaf([]):
#     print(x.print_self())
#     x.population.print_population_simple()
#
# print("---------------------------------MERGE TREE---------------------------")
# binary_tree.select_for_merge("00")
# binary_tree.select_for_merge("01")
#
# binary_tree.get_leaf([])[0].population.chromosomes[0].part_chromosomes[0].genes[0].gene_value = 0.111
#
# print(binary_tree.print_tree())
# for x in binary_tree.get_leaf([]):
#     print(x.print_self())
#     x.population.print_population_simple()
#
# print(binary_tree.print_tree())
# for x in binary_tree.get_leaf([]):
#     print(x.print_self())
#     x.population.print_population_simple()
# print("---------------------------------MERGE TREE---------------------------")
# binary_tree.select_for_merge("01")
#
# print(binary_tree.print_tree())
# for x in binary_tree.get_leaf([]):
#     print(x.print_self())
#     x.population.print_population_simple()

