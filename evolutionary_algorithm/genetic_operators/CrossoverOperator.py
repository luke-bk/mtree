import numpy as np
# Import custom mtree chromosome
from evolutionary_algorithm.chromosome.Chromosome import Chromosome


def crossover(chromosome_one, chromosome_two):
    """
    Perform two-point crossover between two chromosomes.

    Args:
        chromosome_one (Chromosome): The first chromosome.
        chromosome_two (Chromosome): The second chromosome.

    Returns:
        Chromosome: A new chromosome obtained from the crossover.
    """
    # Get the length of the part chromosomes
    part_chromosome_length = len(chromosome_one.part_chromosomes[0].genes)
    # Choose two random crossover points
    crossover_point1 = np.random.randint(0, part_chromosome_length - 1)
    crossover_point2 = np.random.randint(crossover_point1 + 1, part_chromosome_length)

    # Swap genes between the two crossover points in part_chromosomes[0] in both chromosomes
    temp_genes = chromosome_one.part_chromosomes[0].genes[crossover_point1:crossover_point2]
    chromosome_one.part_chromosomes[0].genes[crossover_point1:crossover_point2] = \
        chromosome_two.part_chromosomes[0].genes[crossover_point1:crossover_point2]
    chromosome_two.part_chromosomes[0].genes[crossover_point1:crossover_point2] = temp_genes

    # Get the length of the part chromosomes
    part_chromosome_length = len(chromosome_one.part_chromosomes[1].genes)
    # Choose two random crossover points
    crossover_point1 = np.random.randint(0, part_chromosome_length - 1)
    crossover_point2 = np.random.randint(crossover_point1 + 1, part_chromosome_length)
    # Swap genes between the two crossover points in part_chromosomes[1] in both chromosomes
    temp_genes = chromosome_one.part_chromosomes[1].genes[crossover_point1:crossover_point2]
    chromosome_one.part_chromosomes[1].genes[crossover_point1:crossover_point2] = \
        chromosome_two.part_chromosomes[1].genes[crossover_point1:crossover_point2]
    chromosome_two.part_chromosomes[1].genes[crossover_point1:crossover_point2] = temp_genes

    # Randomly choose one of the part_chromosomes[0]'s and one of the part_chromosomes[1]'s
    new_chromosome = Chromosome(chromosome_one.parent_name, chromosome_one.part_chromosomes_length,
                                chromosome_one.gene_type)

    new_chromosome.part_chromosomes.append(np.random.choice([chromosome_one.part_chromosomes[0],
                                                           chromosome_two.part_chromosomes[0]]))
    new_chromosome.part_chromosomes.append(np.random.choice([chromosome_one.part_chromosomes[1],
                                                           chromosome_two.part_chromosomes[1]]))

    return new_chromosome
