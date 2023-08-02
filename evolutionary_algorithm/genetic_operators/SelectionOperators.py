import numpy as np

from evolutionary_algorithm.chromosome.Chromosome import Chromosome


def sus_selection(chromosomes: Chromosome, num_selected: int) -> list:
    """
    Perform Stochastic Universal Sampling (SUS) selection on a list of chromosomes.

    Args:
        chromosomes (List[Chromosome]): The list of chromosomes.
        num_selected (int): The number of chromosomes to be selected.

    Returns:
        List[Chromosome]: A list of selected chromosomes.
    """
    # Calculate fitness values for all chromosomes
    fitness_values = [chromosome.get_fitness() for chromosome in chromosomes]

    # Calculate total fitness
    total_fitness = sum(fitness_values)

    # Calculate step size
    step_size = total_fitness / num_selected

    # Random starting point on the roulette wheel
    start_point = np.random.uniform(0, step_size)

    # Initialize the selected chromosomes list
    selected_chromosomes = []

    # Spin the roulette wheel to select chromosomes
    current_position = start_point
    for _ in range(num_selected):
        # Find the chromosome corresponding to the current position
        index = 0
        current_sum = fitness_values[0]
        while current_sum < current_position:
            index += 1
            current_sum += fitness_values[index]

        # Add the selected chromosome to the list
        selected_chromosomes.append(chromosomes[index].clone())

        # Move to the next position
        current_position += step_size

    return selected_chromosomes
