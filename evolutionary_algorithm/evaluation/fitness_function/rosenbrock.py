import itertools
from typing import List

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


def rosenbrock(chromosomes: List[Chromosome]) -> float:
    """
    Calculate the value of the Rosenbrock function for a given list or array of n variables.

    Parameters:
    x (list or array): The input list or array containing n variables.

    Returns:
    float: The result of evaluating the Rosenbrock function.

    The Rosenbrock function is defined as:
    f(x) = sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2) for i in range(n-1)
    """
    flat_chromosome = list(itertools.chain.from_iterable(chromosomes))
    x = []
    for chromosome in flat_chromosome:
        x.append(chromosome.express_highest_dominance())
    n = len(x)
    result = 0

    for i in range(n - 1):
        result += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2

    return result
