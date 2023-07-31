import pytest

from evolutionary_algorithm.chromosome.Chromosome import Chromosome
from evolutionary_algorithm.genetic_operators.CrossoverOperator import crossover


@pytest.fixture
def sample_chromosomes():
    # Create sample chromosomes for testing
    parent_name = "Parent"
    part_chromosome_length = 5
    gene_type = "real"
    gene_min = 0.0
    gene_max = 1.0

    chromosome_one = Chromosome(parent_name, part_chromosome_length, gene_type, gene_min, gene_max)
    chromosome_two = Chromosome(parent_name, part_chromosome_length, gene_type, gene_min, gene_max)

    return chromosome_one, chromosome_two


def test_crossover_part_chromosomes(sample_chromosomes):
    chromosome_one, chromosome_two = sample_chromosomes

    # Perform crossover
    new_chromosome = crossover(chromosome_one, chromosome_two)

    # Ensure part_chromosomes[0] and part_chromosomes[1] do not mix
    assert chromosome_one.part_chromosomes[0].genes != new_chromosome.part_chromosomes[0].genes
    assert chromosome_two.part_chromosomes[0].genes != new_chromosome.part_chromosomes[1].genes
    assert chromosome_one.part_chromosomes[1].genes != new_chromosome.part_chromosomes[1].genes
    assert chromosome_two.part_chromosomes[1].genes != new_chromosome.part_chromosomes[0].genes


def test_crossover_clone(sample_chromosomes):
    chromosome_one, chromosome_two = sample_chromosomes

    # Perform crossover
    new_chromosome = crossover(chromosome_one, chromosome_two)

    # Ensure the clone is a deep copy
    assert new_chromosome is not chromosome_one
    assert new_chromosome is not chromosome_two

