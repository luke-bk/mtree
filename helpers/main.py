from evolutionary_algorithm.chromosome.PartChromosome import PartChromosome

# Create a PartChromosome for the binary version
binary_chromosome = PartChromosome("Bit Parent", part_chromosome_length=10, gene_type='bit')

# Split part chromosome
binary_first_half, binary_second_half = binary_chromosome.split_chromosome()

binary_chromosome.print_values()
print(" ---------------- ")
binary_first_half.print_values()
print(" ---------------- ")
binary_second_half.print_values()
print(" --------------------------------------------------------- ")

print(binary_chromosome.genes[0].get_mutation())
binary_chromosome.genes[0].set_mutation(1.0)

print(binary_chromosome.genes[0].get_mutation())
print(binary_first_half.genes[0].get_mutation())
