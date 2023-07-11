import pytest
from evolutionary_algorithm.chromosome.Chromosome import Chromosome


# Test Case 1: Test splitting the chromosome into two halves
def test_chromosome_splitting():
    chromosome = Chromosome("Chromosome", "ParentChromosome", 10, "real", 0.0, 1.0)
    first_half, second_half = chromosome.split_chromosome()

    # Assert the length of the split chromosomes
    assert len(first_half.part_chromosomes[0].genes) == 5
    assert len(first_half.part_chromosomes[1].genes) == 5
    assert len(second_half.part_chromosomes[0].genes) == 5
    assert len(second_half.part_chromosomes[1].genes) == 5

    # Assert the names of the split chromosomes
    assert first_half.name != chromosome.name
    assert second_half.name != chromosome.name
    assert first_half.parent_name == "Chromosome"
    assert second_half.parent_name == "Chromosome"

    # Assert the names of the split part chromosomes
    assert first_half.part_chromosomes[0].name != chromosome.part_chromosomes[0].name
    assert first_half.part_chromosomes[1].name != chromosome.part_chromosomes[1].name
    assert second_half.part_chromosomes[0].name != chromosome.part_chromosomes[0].name
    assert second_half.part_chromosomes[1].name != chromosome.part_chromosomes[1].name


# Test Case 2: Test cloning the chromosome
def test_chromosome_cloning():
    chromosome = Chromosome("Chromosome", "ParentChromosome", 10, "real", 0.0, 1.0)
    clone = chromosome.clone()

    # Assert the clone is a distinct object with a different name
    assert clone is not chromosome
    assert clone.name != chromosome.name

    # Assert the parent names are the same
    assert clone.parent_name == chromosome.parent_name

    # Assert the part chromosomes in the clone are distinct objects with different names
    assert clone.part_chromosomes[0] is not chromosome.part_chromosomes[0]
    assert clone.part_chromosomes[1] is not chromosome.part_chromosomes[1]
    assert clone.part_chromosomes[0].name != chromosome.part_chromosomes[0].name
    assert clone.part_chromosomes[1].name != chromosome.part_chromosomes[1].name


# Test Case 3: Test expressing the highest dominance value for each gene
def test_chromosome_expressing_highest_dominance():
    chromosome = Chromosome("Chromosome", "ParentChromosome", 5, "bit")
    chromosome.part_chromosomes[0].genes[0].set_dominance(0.8)
    chromosome.part_chromosomes[0].genes[1].set_dominance(0.6)
    chromosome.part_chromosomes[0].genes[2].set_dominance(0.7)
    chromosome.part_chromosomes[0].genes[3].set_dominance(0.5)
    chromosome.part_chromosomes[0].genes[4].set_dominance(0.9)

    chromosome.part_chromosomes[1].genes[0].set_dominance(0.7)
    chromosome.part_chromosomes[1].genes[1].set_dominance(0.9)
    chromosome.part_chromosomes[1].genes[2].set_dominance(0.8)
    chromosome.part_chromosomes[1].genes[3].set_dominance(0.4)
    chromosome.part_chromosomes[1].genes[4].set_dominance(0.6)

    expressed_values = chromosome.express_highest_dominance()

    # Assert the values expressed with the highest dominance
    assert expressed_values == [0, 1, 0, 0, 1]


# Test Case 4: Test printing the values of the chromosome
def test_chromosome_printing(capsys):
    chromosome = Chromosome("Chromosome", "ParentChromosome", 5, "bit")
    chromosome.part_chromosomes[0].genes[0].set_gene_value(0)
    chromosome.part_chromosomes[0].genes[1].set_gene_value(1)
    chromosome.part_chromosomes[0].genes[2].set_gene_value(1)
    chromosome.part_chromosomes[0].genes[3].set_gene_value(0)
    chromosome.part_chromosomes[0].genes[4].set_gene_value(1)

    chromosome.part_chromosomes[1].genes[0].set_gene_value(1)
    chromosome.part_chromosomes[1].genes[1].set_gene_value(0)
    chromosome.part_chromosomes[1].genes[2].set_gene_value(0)
    chromosome.part_chromosomes[1].genes[3].set_gene_value(1)
    chromosome.part_chromosomes[1].genes[4].set_gene_value(0)

    chromosome.print_values()

    captured = capsys.readouterr()
    expected_output = """Chromosome Chromosome:
    Part Chromosome 1 ():
        Gene 0: 0
        Gene 1: 1
        Gene 2: 1
        Gene 3: 0
        Gene 4: 1
    Part Chromosome 2 ():
        Gene 0: 1
        Gene 1: 0
        Gene 2: 0
        Gene 3: 1
        Gene 4: 0
"""
    assert captured.out == expected_output


# Run the tests
if __name__ == "__main__":
    pytest.main()
