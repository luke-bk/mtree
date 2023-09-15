# Example usage:

# region = Region(0, 0, 3, 3)
# print(region.get_quadrant(1))  # Prints: [Region (x1=0, y1=1), (x2=1, y2=3)]
#
# point = Point(2, 2)
# print(region.contains_point(point))  # Prints: True
# other_region = Region(2, 2, 4, 4)
# print(region.does_overlap(other_region))  # Prints: True
#
# # Create a Region and a Point
# region = Region(0, 0, 9, 9)
# point = Point(2, 2)
#
# # Create a figure and axis with inverted Y-axis
# fig, ax = plt.subplots(figsize=(5, 5))
# ax.plot([region.get_x1(), region.get_x2(), region.get_x2(), region.get_x1(), region.get_x1()],
#          [region.get_y1(), region.get_y1(), region.get_y2(), region.get_y2(), region.get_y1()], 'b-', label='Region')
# ax.plot(point.get_x(), point.get_y(), 'ro', label='Point')
#
# # Add labels and legend
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_title('Region and Point Visualization')
# ax.legend()
#
# # Invert the Y-axis
# ax.invert_yaxis()
#
# # Set the X-axis tick positions and labels
# x_ticks = np.arange(region.get_x1(), region.get_x2() + 1, 1)
# ax.set_xticks(x_ticks)
# ax.set_xticklabels(x_ticks)
#
# # Move X-axis labels to the top
# ax.xaxis.set_ticks_position('top')
# ax.xaxis.set_label_position('top')
#
# # Set the Y-axis tick positions and labels
# y_ticks = np.arange(region.get_y1(), region.get_y2() + 1, 1)
# ax.set_yticks(y_ticks)
# ax.set_yticklabels(y_ticks)
#
# # Show the plot
# plt.grid(True)
# plt.show()
from evolutionary_algorithm.population.structure.binary_tree.BinaryTree import BinaryTree
from evolutionary_algorithm.population.structure.binary_tree.Point1D import Point1D
from evolutionary_algorithm.population.structure.binary_tree.Region1D import Region1D
from helpers.random_generator import RandomGenerator
from evolutionary_algorithm.population.structure.quad_tree.QuadTree import QuadTree
from evolutionary_algorithm.population.structure.quad_tree.Region import Region
from evolutionary_algorithm.population.Population import Population
from evolutionary_algorithm.chromosome.Chromosome import Chromosome


# # Create a random generator (you can use Python's random module)
# random_generator = RandomGenerator(seed=1)
# # Create an initial (root node) mtree population (where each individual is a list of integers)
# pop = Population("0", 0, 0)
#
# # Populate with randomly generated bit chromosomes, of chromosome_length size
# for _ in range(10):
#     pop.add_chromosome(Chromosome(random_generator, pop.get_name(), 10, "bit"))
#
# # Define the initial region and parameters
# initial_area = Region(0, 0, 23, 23)
# initial_level = 0
# initial_quad = 0
# parent_quadtree = None  # Set to None for the root QuadTree
# child_number = 0
# initial_population = pop
# generation = 0
# best_fitness = 0.0
#
# # Create an instance of the QuadTree
# quadtree = QuadTree(random_generator, initial_area,
#                     initial_level, initial_quad,
#                     parent_quadtree, child_number,
#                     initial_population, generation,
#                     best_fitness)
#
# print(quadtree.print_tree())
#
# # Perform operations on the QuadTree as needed
# quadtree.select_for_split(0, quadtree.pop)
#
# print(quadtree.print_tree())
#
# for x in quadtree.get_leaf([]):
#     print(x.print_tree())
# # quadtree.merge()
#
#

# Import the Point1D and Region1D classes (assuming you have defined them as shown previously)
# from point_and_region_classes import Point1D, Region1D

# # Create some Point1D instances
# point1 = Point1D(2)
# point2 = Point1D(5)
# point3 = Point1D(8)
#
# # Create some Region1D instances
# region1 = Region1D(0, 6)
# region2 = Region1D(4, 10)
# region3 = Region1D(7, 12)
#
# # Test contains_point method of Region1D
# print(f"Region1 contains Point1: {region1.contains_point(point1)}")  # Should be True
# print(f"Region2 contains Point1: {region2.contains_point(point1)}")  # Should be True
# print(f"Region2 contains Point2: {region2.contains_point(point2)}")  # Should be True
# print(f"Region3 contains Point2: {region3.contains_point(point2)}")  # Should be True
# print(f"Region1 contains Point3: {region1.contains_point(point3)}")  # Should be False
# print(f"Region3 contains Point3: {region3.contains_point(point3)}")  # Should be True
#
# # Test does_overlap method of Region1D
# print(f"Region1 overlaps with Region2: {region1.does_overlap(region2)}")  # Should be True
# print(f"Region2 overlaps with Region3: {region2.does_overlap(region3)}")  # Should be True
# print(f"Region1 overlaps with Region3: {region1.does_overlap(region3)}")  # Should be False


from evolutionary_algorithm.population.structure.binary_tree.Point1D import Point1D
from evolutionary_algorithm.population.structure.binary_tree.Region1D import Region1D
from helpers.random_generator import RandomGenerator
from evolutionary_algorithm.population.structure.quad_tree.QuadTree import QuadTree
from evolutionary_algorithm.population.structure.quad_tree.Region import Region
from evolutionary_algorithm.population.Population import Population
from evolutionary_algorithm.chromosome.Chromosome import Chromosome

# Create a random generator
random_generator = RandomGenerator(seed=1)

gen = 0
# Create a root node
root_region = Region1D(0, 23)
root_pop = Population(random_generator=random_generator, name="0", generation=gen, fitness=0, parent_population=None)
for _ in range(4):
    root_pop.add_chromosome(Chromosome(random_generator, root_pop.get_name(), 8, "bit"))

for x in root_pop.chromosomes:
    x.set_fitness(1)

binary_tree = BinaryTree(random_generator=random_generator, region=root_region,
                         level=0, parent=None,
                         child_number=0, population=root_pop,
                         max_depth=3)

print("---------------------------------TREE CREATED---------------------------")
print(binary_tree.print_tree())
for x in binary_tree.get_leaf([]):
    print(x.print_self())
    x.population.print_population_simple()

print("---------------------------------SPLIT TREE---------------------------")
binary_tree.select_for_split(0)

print(binary_tree.print_tree())
for x in binary_tree.get_leaf([]):
    print(x.print_self())
    x.population.print_population_simple()

print("---------------------------------SPLIT TREE---------------------------")
binary_tree.select_for_split(0)

print(binary_tree.print_tree())
for x in binary_tree.get_leaf([]):
    print(x.print_self())
    x.population.print_population_simple()

print("---------------------------------MERGE TREE---------------------------")
binary_tree.select_for_merge("010")
binary_tree.select_for_merge("011")

print(binary_tree.print_tree())
for x in binary_tree.get_leaf([]):
    print(x.print_self())
    x.population.print_population_simple()

print("---------------------------------MERGE TREE---------------------------")
binary_tree.select_for_merge("00")

print(binary_tree.print_tree())
for x in binary_tree.get_leaf([]):
    print(x.print_self())
    x.population.print_population_simple()
print("---------------------------------MERGE TREE---------------------------")
binary_tree.select_for_merge("01")

print(binary_tree.print_tree())
for x in binary_tree.get_leaf([]):
    print(x.print_self())
    x.population.print_population_simple()

