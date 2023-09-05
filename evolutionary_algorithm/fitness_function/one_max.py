from evolutionary_algorithm.chromosome.Chromosome import Chromosome


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
