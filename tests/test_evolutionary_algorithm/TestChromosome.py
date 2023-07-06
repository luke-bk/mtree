import pytest
from evolutionary_algorithm.Chromosome import Chromosome


def test_equivalent_partitioning():
    # Test Case 1
    chromosome1 = Chromosome("Chromosome1", 10, 0.0, 1.0)
    assert chromosome1.name == "Chromosome1"
    assert chromosome1.get_chromosome_length() == 10

    # Test Case 2
    chromosome2 = Chromosome("Chromosome2", 5, -10.0, 10.0)
    assert chromosome2.name == "Chromosome2"
    assert chromosome2.get_chromosome_length() == 5

    # Test Case 3: Negative length chromosome
    with pytest.raises(ValueError):
        chromosome3 = Chromosome("Chromosome3", -5, 0.0, 1.0)

    # Test Case 4: Chromosome of length 0
    with pytest.raises(ValueError):
        chromosome4 = Chromosome("Chromosome4", 0, 0.0, 1.0)


def test_boundary_value_analysis():
    # Test Case 5: Minimum allowed length (length = 1)
    chromosome5 = Chromosome("Chromosome5", 1, 0.0, 1.0)
    assert chromosome5.name == "Chromosome5"
    assert chromosome5.get_chromosome_length() == 1

    # Test Case 6: Length greater than the minimum (length = 2)
    chromosome6 = Chromosome("Chromosome6", 2, -10.0, 10.0)
    assert chromosome6.name == "Chromosome6"
    assert chromosome6.get_chromosome_length() == 2

    # Test Case 7: Minimum allowed gene values
    chromosome7 = Chromosome("Chromosome7", 10, -1e9, -1e9)
    assert chromosome7.name == "Chromosome7"
    assert chromosome7.get_chromosome_length() == 10
