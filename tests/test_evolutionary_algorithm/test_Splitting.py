from random import Random
import pytest
from evolutionary_algorithm.population.Population import Population  # Replace 'your_module' with the actual module where your Population class is defined

@pytest.fixture
def random_generator():
    return Random(42)  # Seed the random generator for reproducibility

@pytest.fixture
def test_population(random_generator):
    return Population(random_generator, "Root", 0, 0)  # Create a population for testing

def test_population_splitting(test_population):
    # Add some chromosomes to the population
    for i in range(10):
        chromosome = your_chromosome_class()  # Replace with your chromosome creation logic
        test_population.add_chromosome(chromosome)

    # Split the population
    generation = 1
    child1, child2 = test_population.split_population(generation)

    # Check if the split occurred correctly
    assert len(child1.chromosomes) + len(child2.chromosomes) == len(test_population.chromosomes)
    assert child1.generation == generation
    assert child2.generation == generation
    assert child1.parent_population == test_population
    assert child2.parent_population == test_population

if __name__ == '__main__':
    pytest.main()
