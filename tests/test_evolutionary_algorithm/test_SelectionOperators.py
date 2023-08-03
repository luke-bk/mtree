import pytest
import numpy as np

from evolutionary_algorithm.chromosome.Chromosome import Chromosome
from evolutionary_algorithm.genetic_operators.SelectionOperators import sus_selection


# Mock fitness function for testing
def mock_fitness_function(chromosome):
    return chromosome.get_fitness()


# Helper function to create a list of mock chromosomes with random fitness values
def create_mock_chromosomes(num_chromosomes):
    chromosomes = []
    for _ in range(num_chromosomes):
        chromosome = Chromosome("Parent", 10, "bit")
        chromosome.set_fitness(np.random.uniform(0, 1))  # Assign a random fitness value
        chromosomes.append(chromosome)
    return chromosomes


# Test SUS selection returns the correct number of selected chromosomes
def test_sus_selection_returns_correct_number():
    num_selected = 5
    chromosomes = create_mock_chromosomes(10)
    selected_chromosomes = sus_selection(chromosomes, num_selected)
    assert len(selected_chromosomes) == num_selected


# Test SUS selection selects chromosomes with higher fitness more frequently
def test_sus_selection_prefers_higher_fitness_chromosomes():
    num_selected = 5
    chromosomes = create_mock_chromosomes(10)
    selected_chromosomes = sus_selection(chromosomes, num_selected)

    # Calculate the average fitness of the selected chromosomes
    average_fitness_selected = sum(
        mock_fitness_function(chromosome) for chromosome in selected_chromosomes) / num_selected

    # Calculate the average fitness of the original chromosomes
    average_fitness_original = sum(mock_fitness_function(chromosome) for chromosome in chromosomes) / len(chromosomes)

    # The average fitness of selected chromosomes should be higher than the average fitness of original chromosomes
    assert average_fitness_selected > average_fitness_original


# Test SUS selection raises ValueError if num_selected is greater than the number of chromosomes
def test_sus_selection_raises_value_error():
    num_selected = 20
    chromosomes = create_mock_chromosomes(10)
    with pytest.raises(ValueError):
        sus_selection(chromosomes, num_selected)


# Test SUS selection does not modify the original chromosomes
def test_sus_selection_does_not_modify_original_chromosomes():
    num_selected = 5
    chromosomes = create_mock_chromosomes(10)
    original_chromosomes = [chromosome.clone() for chromosome in
                            chromosomes]  # Create deep copies of original chromosomes
    sus_selection(chromosomes, num_selected)

    # Check if the genes of the original chromosomes are not modified
    for original_chromosome, chromosome in zip(original_chromosomes, chromosomes):
        assert original_chromosome.part_chromosomes[0].genes == chromosome.part_chromosomes[0].genes
        assert original_chromosome.part_chromosomes[1].genes == chromosome.part_chromosomes[1].genes
