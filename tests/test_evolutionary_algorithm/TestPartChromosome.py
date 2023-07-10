import pytest

from evolutionary_algorithm.gene.MtreeGene import MtreeGene
from evolutionary_algorithm.chromosome.PartChromosome import PartChromosome


@pytest.fixture
def part_chromosome():
    return PartChromosome(10, 0.0, 1.0)


def test_chromosome_length(part_chromosome):
    """
    Test case to verify the length of the chromosome.
    """
    assert part_chromosome.get_chromosome_length() == 10


def test_get_gene(part_chromosome):
    """
    Test case to verify retrieving a gene from the chromosome.
    """
    gene = part_chromosome.get_gene(0)
    assert isinstance(gene, MtreeGene)


def test_set_gene(part_chromosome):
    """
    Test case to verify setting a gene in the chromosome.
    """
    gene = MtreeGene(0.0, 1.0)
    part_chromosome.set_gene(0, gene)
    assert part_chromosome.get_gene(0) == gene


def test_split_chromosome(part_chromosome):
    """
    Test case to verify splitting the chromosome into two halves.
    """
    first_half, second_half = part_chromosome.split_chromosome()
    assert isinstance(first_half, list)
    assert isinstance(second_half, list)
    assert len(first_half) == 5
    assert len(second_half) == 5


def test_split_chromosome_odd_length():
    """
    Test case to verify splitting the chromosome with odd length.
    """
    part_chromosome = PartChromosome(11, 0.0, 1.0)
    first_half, second_half = part_chromosome.split_chromosome()
    assert len(first_half) == 6
    assert len(second_half) == 5


def test_split_chromosome_deep_copy(part_chromosome):
    """
    Test case to verify the deep copy behavior during chromosome splitting.
    """
    first_half, second_half = part_chromosome.split_chromosome()
    first_half[0].set_gene_value(2.0)
    assert part_chromosome.get_gene(0).get_gene_value() != 2.0
