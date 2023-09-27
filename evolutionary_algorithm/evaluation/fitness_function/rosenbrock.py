import itertools
from typing import List
from helpers.utilities import flatten_chromosomes

from evolutionary_algorithm.chromosome.Chromosome import Chromosome


def rosenbrock(x) -> float:
    """
    Calculate the value of the Rosenbrock function for a given list or array of n variables.

    Parameters:
    x (list or array): The input list or array containing n variables.

    Returns:
    float: The result of evaluating the Rosenbrock function.

    The Rosenbrock function is defined as:
    f(x) = sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2) for i in range(n-1)
    """
    n = len(x)
    result = 0

    for i in range(n - 1):
        result += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2

    return result


def rosenbrock_mtree(context_vector: List[Chromosome]) -> float:
    """
    Calculate the value of the Rosenbrock function for a given list or array of n variables.

    Parameters:
    context_vector (list or array): The input list or array containing n variables.

    Returns:
    float: The result of evaluating the Rosenbrock function.

    The Rosenbrock function is defined as:
    f(expressed_genes) = sum(100 * (expressed_genes[i+1] - expressed_genes[i]**2)**2 + (1 - expressed_genes[i])**2)
    for i in range(n-1)
    """
    # Check if context_vector is a list of lists, if so, flatten and return a single list
    flat_context_vector = flatten_chromosomes(context_vector)

    expressed_genes = []
    for chromosome in flat_context_vector:
        expressed_genes = expressed_genes + chromosome.express_highest_dominance()

    n = len(expressed_genes)
    result = 0

    for i in range(n - 1):
        result += 100 * (expressed_genes[i + 1] - expressed_genes[i] ** 2) ** 2 + (1 - expressed_genes[i]) ** 2

    return result
