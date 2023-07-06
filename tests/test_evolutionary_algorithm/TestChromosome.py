import pytest
from evolutionary_algorithm.Chromosome import Chromosome


def test_equivalent_partitioning():
    # Test Case 1
    chromosome1 = Chromosome(10, 0.0, 1.0)
    assert chromosome1.name == hex(id(chromosome1))
    assert chromosome1.get_chromosome_length() == 10

    # Test Case 2
    chromosome2 = Chromosome(5, -10.0, 10.0)
    assert chromosome2.name == hex(id(chromosome2))
    assert chromosome2.get_chromosome_length() == 5


def test_boundary_value_analysis():
    # Test Case 5: Minimum allowed length (length = 1)
    chromosome5 = Chromosome(1, 0.0, 1.0)
    assert chromosome5.name == hex(id(chromosome5))
    assert chromosome5.get_chromosome_length() == 1

    # Test Case 6: Length greater than the minimum (length = 2)
    chromosome6 = Chromosome(2, -10.0, 10.0)
    assert chromosome6.name == hex(id(chromosome6))
    assert chromosome6.get_chromosome_length() == 2

    # Test Case 7: Minimum allowed gene values
    chromosome7 = Chromosome(10, -1e9, -1e9)
    assert chromosome7.name == hex(id(chromosome7))
    assert chromosome7.get_chromosome_length() == 10
