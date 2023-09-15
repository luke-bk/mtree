from evolutionary_algorithm.chromosome.Chromosome import Chromosome
from typing import List


def fitness_function(chromosome: Chromosome) -> float:
    """
    Calculate the fitness of a chromosome based on the sum of the highest dominance genes.

    Args:
        chromosome (Chromosome): The chromosome to evaluate.

    Returns:
        float: The fitness value, which is the sum of the highest dominance genes' values.
    """
    # Calculate the fitness by summing the values of the highest dominance genes in the chromosome.
    return sum(chromosome.express_highest_dominance())


def fitness_function_mtree(chromosomes: List[Chromosome]) -> float:
    """
    Calculate the fitness of an array of chromosomes based on the sum of the highest dominance genes for each chromosome.

    Args:
        chromosomes (List[Chromosome]): The list of chromosomes to evaluate.

    Returns:
        float: The fitness value, which is the sum of the highest dominance genes' values for each chromosome.
    """
    # Initialize the total fitness value to zero.
    total_fitness = 0.0

    # Loop through each chromosome in the array.
    for chromosome in chromosomes:
        # Calculate the fitness for the current chromosome and add it to the total fitness.
        total_fitness += sum(chromosome.express_highest_dominance())

    return total_fitness
