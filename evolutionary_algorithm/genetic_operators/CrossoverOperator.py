

def crossover(random_generator, chromosome_one, chromosome_two):
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
    crossover_point1 = random_generator.randint(0, part_chromosome_length - 1)
    crossover_point2 = random_generator.randint(crossover_point1 + 1, part_chromosome_length)

    # Swap genes between the two crossover points in part_chromosomes[0] in both chromosomes
    temp_genes = chromosome_one.part_chromosomes[0].genes[crossover_point1:crossover_point2]
    chromosome_one.part_chromosomes[0].genes[crossover_point1:crossover_point2] = \
        chromosome_two.part_chromosomes[0].genes[crossover_point1:crossover_point2]
    chromosome_two.part_chromosomes[0].genes[crossover_point1:crossover_point2] = temp_genes

    # Get the length of the part chromosomes
    part_chromosome_length = len(chromosome_one.part_chromosomes[1].genes)
    # Choose two random crossover points
    crossover_point1 = random_generator.randint(0, part_chromosome_length - 1)
    crossover_point2 = random_generator.randint(crossover_point1 + 1, part_chromosome_length)
    # Swap genes between the two crossover points in part_chromosomes[1] in both chromosomes
    temp_genes = chromosome_one.part_chromosomes[1].genes[crossover_point1:crossover_point2]
    chromosome_one.part_chromosomes[1].genes[crossover_point1:crossover_point2] = \
        chromosome_two.part_chromosomes[1].genes[crossover_point1:crossover_point2]
    chromosome_two.part_chromosomes[1].genes[crossover_point1:crossover_point2] = temp_genes


def crossover_part_chromosomes_in_place(random_generator, chromosome_one, chromosome_two):
    """
    Perform two-point crossover between two chromosomes in place.

    Args:
        chromosome_one (Chromosome): The first chromosome.
        chromosome_two (Chromosome): The second chromosome.
    """
    # Get the length of the part chromosomes
    part_chromosome_length = len(chromosome_one.part_chromosomes[0].genes)
    # Choose two random crossover points
    crossover_point1 = random_generator.randint(0, part_chromosome_length - 1)
    crossover_point2 = random_generator.randint(crossover_point1 + 1, part_chromosome_length)

    # Swap genes between the two crossover points in part_chromosomes[0] in both chromosomes
    temp_genes = chromosome_one.part_chromosomes[0].genes[crossover_point1:crossover_point2].copy()
    chromosome_one.part_chromosomes[0].genes[crossover_point1:crossover_point2] = \
        chromosome_two.part_chromosomes[0].genes[crossover_point1:crossover_point2]
    chromosome_two.part_chromosomes[0].genes[crossover_point1:crossover_point2] = temp_genes

    # Get the length of the part chromosomes
    part_chromosome_length = len(chromosome_one.part_chromosomes[1].genes)
    # Choose two random crossover points
    crossover_point1 = random_generator.randint(0, part_chromosome_length - 1)
    crossover_point2 = random_generator.randint(crossover_point1 + 1, part_chromosome_length)
    # Swap genes between the two crossover points in part_chromosomes[1] in both chromosomes
    temp_genes = chromosome_one.part_chromosomes[1].genes[crossover_point1:crossover_point2].copy()
    chromosome_one.part_chromosomes[1].genes[crossover_point1:crossover_point2] = \
        chromosome_two.part_chromosomes[1].genes[crossover_point1:crossover_point2]
    chromosome_two.part_chromosomes[1].genes[crossover_point1:crossover_point2] = temp_genes
