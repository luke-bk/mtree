from evolutionary_algorithm.chromosome.Chromosome import Chromosome


def sus_selection(random_generator, chromosomes: Chromosome, num_selected: int) -> list:
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
    start_point = random_generator.uniform(0, step_size)

    # Initialize the selected chromosomes list
    selected_chromosomes = []

    # Track the previous chromosome instance
    previous_chromosome = None

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


def sus_selection_fast_clone(random_generator, chromosomes: Chromosome, num_selected: int) -> list:
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
    start_point = random_generator.uniform(0, step_size)

    # Initialize the selected chromosomes list
    selected_chromosomes = []

    # Track the previous chromosome instance
    previous_chromosome = None

    # Spin the roulette wheel to select chromosomes
    current_position = start_point
    for _ in range(num_selected):
        # Find the chromosome corresponding to the current position
        index = 0
        current_sum = fitness_values[0]
        while current_sum < current_position:
            index += 1
            current_sum += fitness_values[index]

        # Get the selected chromosome
        selected_chromosome = chromosomes[index]

        # Check if it's the same instance as the previous one
        if selected_chromosome is previous_chromosome:
            selected_chromosome = selected_chromosome.clone()

        # Add the selected chromosome to the list
        selected_chromosomes.append(selected_chromosome)

        # Update the previous chromosome reference
        previous_chromosome = selected_chromosome

        # Move to the next position
        current_position += step_size

    return selected_chromosomes


def tournament_selection(random_generator, population, tournament_size, num_selected, is_minimization_task):
    """
    Perform Tournament Selection with duplicate cloning on a population of chromosomes.

    Args:
        random_generator (Random): A random number generator for consistent randomness.
        population (List[Chromosome]): The list of chromosomes to select from.
        tournament_size (int): The size of the tournament (number of randomly selected individuals).
        num_selected (int): The number of chromosomes to be selected.
        is_minimization_task (boolean): Min or max fitness better?

    Returns:
        List[Chromosome]: A list of selected chromosomes, with duplicates cloned if needed.
    """
    # Initialize a list to store the selected chromosomes
    selected_chromosomes = []

    # Repeat the selection process for the specified number of times
    for _ in range(num_selected):
        # Randomly select a subset of individuals for the tournament
        tournament = random_generator.choice(population, tournament_size)

        # Sort the tournament based on fitness, assuming lower fitness is better
        tournament = sorted(tournament, key=lambda ind: ind.get_fitness())

        # Select the best individual from the tournament
        selected_chromosome = tournament[0] if is_minimization_task else tournament[tournament_size - 1]

        # Check if the selected chromosome is a duplicate (already in the selection)
        if selected_chromosome in selected_chromosomes:
            # If duplicate, create a clone to avoid modifying the original
            selected_chromosome = selected_chromosome.clone()

        # Add the selected chromosome to the list of selected individuals
        selected_chromosomes.append(selected_chromosome)

    # Return the list of selected chromosomes
    return selected_chromosomes
