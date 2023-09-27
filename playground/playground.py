# Example usage:
from helpers.utilities import flatten_chromosomes

chromosomes1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat_chromosome1 = flatten_chromosomes(chromosomes1)
print(flat_chromosome1)

chromosomes2 = [1, 2, 3, 4, 5]
flat_chromosome2 = flatten_chromosomes(chromosomes2)
print(flat_chromosome2)

chromosomes3 = "Not a list"
flat_chromosome3 = flatten_chromosomes(chromosomes3)
print(flat_chromosome3)
