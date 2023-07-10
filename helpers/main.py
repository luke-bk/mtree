from evolutionary_algorithm.chromosome.PartChromosome import PartChromosome

# Create a PartChromosome for the binary version
binary_chromosome = PartChromosome("Bit Parent", part_chromosome_length=10, gene_type='bit')

# Print each gene value in the binary chromosome
print("Binary Chromosome:")
for i in range(binary_chromosome.get_chromosome_length()):
    gene = binary_chromosome.get_gene(i)
    print(f"Gene {i+1}: {gene.get_gene_value()}")

# Create a PartChromosome for the real version
real_chromosome = PartChromosome("Real Parent", part_chromosome_length=10, gene_type='real', gene_min=0.0, gene_max=1.0)

# Print each gene value in the real chromosome
print("\nReal Chromosome:")
for i in range(real_chromosome.get_chromosome_length()):
    gene = real_chromosome.get_gene(i)
    print(f"Gene {i+1}: {gene.get_gene_value()}")

# Print dominance and mutation values for both chromosomes
print("\nDominance and Mutation Values:")
for i in range(binary_chromosome.get_chromosome_length()):
    binary_gene = binary_chromosome.get_gene(i)
    real_gene = real_chromosome.get_gene(i)
    print(f"Gene {i+1}: Binary Dominance - {binary_gene.get_dominance()}, Binary Mutation - {binary_gene.get_mutation()}")
    print(f"         Real Dominance - {real_gene.get_dominance()}, Real Mutation - {real_gene.get_mutation()}")

# Split both part chromosomes
binary_first_half, binary_second_half = binary_chromosome.split_chromosome()
real_first_half, real_second_half = real_chromosome.split_chromosome()

# Deep copy the first half of binary chromosome
binary_first_half_copy = binary_first_half._deep_copy_chromosome(binary_first_half)

# Change the gene value of the copied binary chromosome
binary_first_half_copy[0].set_gene_value(1.0)

# Verify the changes in the copied binary chromosome
print("\nCopied Binary Chromosome:")
for i in range(len(binary_first_half_copy)):
    gene = binary_first_half_copy[i]
    print(f"Gene {i+1}: {gene.get_gene_value()}")

# Verify the original binary chromosome remains unchanged
print("\nOriginal Binary Chromosome:")
for i in range(len(binary_first_half)):
    gene = binary_first_half[i]
    print(f"Gene {i+1}: {gene.get_gene_value()}")
