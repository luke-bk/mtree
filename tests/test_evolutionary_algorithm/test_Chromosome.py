import pytest
from evolutionary_algorithm.chromosome.Chromosome import Chromosome


# Test Case 1: Test splitting the chromosome into two halves
def test_chromosome_splitting():
    chromosome = Chromosome("ParentChromosome", 10, "real", 0.0, 1.0)
    first_half, second_half = chromosome.split_chromosome()

    # Assert the length of the split chromosomes
    assert len(first_half.part_chromosomes[0].genes) == 5
    assert len(first_half.part_chromosomes[1].genes) == 5
    assert len(second_half.part_chromosomes[0].genes) == 5
    assert len(second_half.part_chromosomes[1].genes) == 5

    # Assert the names of the split chromosomes
    assert first_half.name != chromosome.name
    assert second_half.name != chromosome.name
    assert first_half.parent_name == "ParentChromosome"
    assert second_half.parent_name == "ParentChromosome"

    # Assert the names of the split part chromosomes
    assert first_half.part_chromosomes[0].name != chromosome.part_chromosomes[0].name
    assert first_half.part_chromosomes[1].name != chromosome.part_chromosomes[1].name
    assert second_half.part_chromosomes[0].name != chromosome.part_chromosomes[0].name
    assert second_half.part_chromosomes[1].name != chromosome.part_chromosomes[1].name


# Test Case 2: Test cloning the chromosome
def test_chromosome_cloning():
    chromosome = Chromosome("ParentChromosome", 10, "real", 0.0, 1.0)
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
    chromosome = Chromosome("ParentChromosome", 5, "bit")
    #  All part 0 chromosomes have the highest dom values
    chromosome.part_chromosomes[0].genes[0].set_dominance(0.9)
    chromosome.part_chromosomes[0].genes[1].set_dominance(0.9)
    chromosome.part_chromosomes[0].genes[2].set_dominance(0.9)
    chromosome.part_chromosomes[0].genes[3].set_dominance(0.9)
    chromosome.part_chromosomes[0].genes[4].set_dominance(0.9)
    #  All part 1 chromosomes have the lowest dom values
    chromosome.part_chromosomes[1].genes[0].set_dominance(0.1)
    chromosome.part_chromosomes[1].genes[1].set_dominance(0.1)
    chromosome.part_chromosomes[1].genes[2].set_dominance(0.1)
    chromosome.part_chromosomes[1].genes[3].set_dominance(0.1)
    chromosome.part_chromosomes[1].genes[4].set_dominance(0.1)
    #  Set all part 0 chrom values to 1
    chromosome.part_chromosomes[0].genes[0].set_gene_value(1)
    chromosome.part_chromosomes[0].genes[1].set_gene_value(1)
    chromosome.part_chromosomes[0].genes[2].set_gene_value(1)
    chromosome.part_chromosomes[0].genes[3].set_gene_value(1)
    chromosome.part_chromosomes[0].genes[4].set_gene_value(1)
    #  Capture all the expressed values, should be 1
    expressed_values = chromosome.express_highest_dominance()

    # Assert the values expressed with the highest dominance
    assert expressed_values == [1, 1, 1, 1, 1]


# Test Case 4: Test printing the values of the chromosome
def test_chromosome_printing(capsys):
    chromosome = Chromosome("ParentChromosome", 3, "bit")
    chromosome.part_chromosomes[0].genes[0].set_gene_value(0)
    chromosome.part_chromosomes[0].genes[1].set_gene_value(1)
    chromosome.part_chromosomes[0].genes[2].set_gene_value(1)

    chromosome.part_chromosomes[1].genes[0].set_gene_value(1)
    chromosome.part_chromosomes[1].genes[1].set_gene_value(0)
    chromosome.part_chromosomes[1].genes[2].set_gene_value(0)

    chromosome.print_values()

    captured = capsys.readouterr()
    expected_output = f"""Chromosome {chromosome.name}:
    Part Chromosome 1 ({chromosome.part_chromosomes[0].name}):
        Gene 0: 0
        Gene 1: 1
        Gene 2: 1
    Part Chromosome 2 ({chromosome.part_chromosomes[1].name}):
        Gene 0: 1
        Gene 1: 0
        Gene 2: 0
"""
    assert captured.out == expected_output


# Run the tests
if __name__ == "__main__":
    pytest.main()
