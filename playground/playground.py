import random
from copy import deepcopy


def tournament_selection(population, tournament_size, num_selected):
    selected_individuals = []

    for _ in range(num_selected):
        # Randomly select individuals for the tournament
        tournament = random.sample(population, tournament_size)

        # Sort the tournament based on fitness (assuming lower fitness is better)
        tournament.sort(key=lambda ind: ind['fitness'])

        # Select the best individual from the tournament
        best_individual = tournament[0]

        # Check if the selected individual is a duplicate
        if tournament.count(best_individual) > 1:
            # Clone the selected individual to avoid modifying the original
            best_individual = deepcopy(best_individual)

        selected_individuals.append(best_individual)

    return selected_individuals


# Example usage:
if __name__ == "__main__":
    # Sample population with fitness values
    population = [{'fitness': 10}, {'fitness': 5}, {'fitness': 5}, {'fitness': 3}]

    # Number of individuals to select
    num_selected = 4

    # Tournament size
    tournament_size = 3

    selected_individuals = tournament_selection(population, tournament_size, num_selected)

    print("Selected Individuals:")
    for individual in selected_individuals:
        print("Fitness:", individual['fitness'])
