

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
    # If the chromosome is greater than one, we can crossover
    if part_chromosome_length > 1:
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


def crossover_image(random_generator, chromosome_one, chromosome_two):
    """
    Perform two-point crossover between two chromosomes.

    Args:
        chromosome_one (Chromosome): The first chromosome.
        chromosome_two (Chromosome): The second chromosome.

    Returns:
        Chromosome: A new chromosome obtained from the crossover.
    """
    if chromosome_one.chromosome.shape != chromosome_two.chromosome.shape:
        raise ValueError("Parent images must have the same shape.")

    # Unpack the two dimensions (grayscale images are 2D)
    rows, cols = chromosome_one.chromosome.shape

    # Randomly choose two points for crossover
    point1 = (random_generator.randint(0, rows), random_generator.randint(0, cols))
    point2 = (random_generator.randint(0, rows), random_generator.randint(0, cols))

    # Ensure that point1 is top left and point2 is bottom right
    start_row = min(point1[0], point2[0])
    end_row = max(point1[0], point2[0])
    start_col = min(point1[1], point2[1])
    end_col = max(point1[1], point2[1])

    # Perform crossover directly on the parents
    temp_region = chromosome_one.chromosome[start_row:end_row, start_col:end_col].copy()
    chromosome_one.chromosome[start_row:end_row, start_col:end_col] = chromosome_two.chromosome[start_row:end_row, start_col:end_col]
    chromosome_two.chromosome[start_row:end_row, start_col:end_col] = temp_region

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
