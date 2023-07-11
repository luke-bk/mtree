import pytest
from evolutionary_algorithm.gene.MtreeGeneBit import MtreeGeneBit
from evolutionary_algorithm.gene.MtreeGeneReal import MtreeGeneReal
from evolutionary_algorithm.chromosome.PartChromosome import PartChromosome


# Test Case 1: Test PartChromosome initialization and getter methods
def test_part_chromosome_initialization():
    parent_name = "ParentChromosome"
    part_chromosome_length = 10
    gene_type = "real"
    gene_min = 0.0
    gene_max = 1.0

    chromosome = PartChromosome(parent_name, part_chromosome_length, gene_type, gene_min, gene_max)

    assert chromosome.name != ""
    assert chromosome.parent_name == parent_name
    assert len(chromosome.genes) == part_chromosome_length

    for i in range(part_chromosome_length):
        gene = chromosome.get_gene(i)
        assert isinstance(gene, MtreeGeneReal)
        assert gene_min <= gene.get_gene_value() <= gene_max


# Test Case 2: Test PartChromosome setter methods
def test_part_chromosome_setter_methods():
    parent_name = "ParentChromosome"
    part_chromosome_length = 10
    gene_type = "bit"

    chromosome = PartChromosome(parent_name, part_chromosome_length, gene_type)

    # Set gene at index 0
    new_gene = MtreeGeneBit()
    chromosome.set_gene(0, new_gene)
    assert chromosome.get_gene(0) is new_gene


# Test Case 3: Test PartChromosome cloning
def test_part_chromosome_cloning():
    parent_name = "ParentChromosome"
    part_chromosome_length = 10
    gene_type = "real"
    gene_min = 0.0
    gene_max = 1.0

    chromosome = PartChromosome(parent_name, part_chromosome_length, gene_type, gene_min, gene_max)
    clone = chromosome._deep_copy_part_chromosome(chromosome.genes)

    assert clone is not chromosome
    assert clone.name != chromosome.name
    assert clone.parent_name == chromosome.parent_name
    assert len(clone.genes) == len(chromosome.genes)

    for i in range(part_chromosome_length):
        clone_gene = clone.get_gene(i)
        original_gene = chromosome.get_gene(i)
        assert clone_gene is not original_gene
        assert clone_gene.get_gene_value() == original_gene.get_gene_value()

    # Change a gene value in the clone and ensure it is different from the original
    clone.set_gene(0, MtreeGeneReal(gene_min, gene_max))
    assert clone.get_gene(0).get_gene_value() != chromosome.get_gene(0).get_gene_value()


# Test Case 4: Test PartChromosome splitting
def test_part_chromosome_splitting():
    parent_name = "ParentChromosome"
    part_chromosome_length = 10
    gene_type = "real"
    gene_min = 0.0
    gene_max = 1.0

    chromosome = PartChromosome(parent_name, part_chromosome_length, gene_type, gene_min, gene_max)
    first_half, second_half = chromosome.split_chromosome()

    assert isinstance(first_half, PartChromosome)
    assert isinstance(second_half, PartChromosome)
    assert first_half is not second_half
    assert first_half.parent_name == chromosome.parent_name
    assert second_half.parent_name == chromosome.parent_name

    assert len(first_half.genes) + len(second_half.genes) == part_chromosome_length


# Test Case 5: Test splitting with odd numbered list
def test_part_chromosome_splitting_odd():
    parent_name = "ParentChromosome"
    part_chromosome_length = 11
    gene_type = "bit"

    chromosome = PartChromosome(parent_name, part_chromosome_length, gene_type)
    first_half, second_half = chromosome.split_chromosome()

    assert len(first_half.genes) + len(second_half.genes) == part_chromosome_length
    assert len(first_half.genes) == (part_chromosome_length // 2) + 1
    assert len(second_half.genes) == part_chromosome_length // 2


# Test Case 6: Test the split chromosomes
def test_part_chromosome_splitting_even_correct():
    # Create a PartChromosome for the binary version
    binary_chromosome = PartChromosome("Bit Parent", part_chromosome_length=10, gene_type='bit')

    # Split part chromosome
    binary_first_half, binary_second_half = binary_chromosome.split_chromosome()

    assert binary_chromosome.genes[0].get_gene_value() == binary_first_half.genes[0].get_gene_value()  # First item in original list is sthe same as first item the first half of the split list
    assert binary_chromosome.genes[4].get_gene_value() == binary_first_half.genes[-1].get_gene_value()  # middle item in original list is sthe same as last item the first half of the split list
    assert binary_chromosome.genes[5].get_gene_value() == binary_second_half.genes[0].get_gene_value()  # one after middle item in original list is the same as first item the second half of the split list
    assert binary_chromosome.genes[-1].get_gene_value() == binary_second_half.genes[-1].get_gene_value()  # last in original list is the same as last item the second half of the split list


# Test Case 7: Test the split chromosomes
def test_part_chromosome_splitting_odd_correct():
    # Create a PartChromosome for the binary version
    binary_chromosome = PartChromosome("Bit Parent", part_chromosome_length=11, gene_type='bit')

    # Split part chromosome
    binary_first_half, binary_second_half = binary_chromosome.split_chromosome()

    assert binary_chromosome.genes[0].get_gene_value() == binary_first_half.genes[0].get_gene_value()  # First item in original list is sthe same as first item the first half of the split list
    assert binary_chromosome.genes[5].get_gene_value() == binary_first_half.genes[-1].get_gene_value()  # middle item in original list is sthe same as last item the first half of the split list
    assert binary_chromosome.genes[6].get_gene_value() == binary_second_half.genes[0].get_gene_value()  # one after middle item in original list is the same as first item the second half of the split list
    assert binary_chromosome.genes[-1].get_gene_value() == binary_second_half.genes[-1].get_gene_value()  # last in original list is the same as last item the second half of the split list


# Test Case 8: Test the split chromosomes and deep clone
def test_part_chromosome_splitting_and_cloning():
    # Create a PartChromosome for the binary version
    binary_chromosome = PartChromosome("Bit Parent", part_chromosome_length=11, gene_type='bit')

    # Split part chromosome
    binary_first_half, binary_second_half = binary_chromosome.split_chromosome()
    assert binary_chromosome.genes[0].get_mutation() == binary_first_half.genes[0].get_mutation()  # Should have the same mutation rate
    binary_chromosome.genes[0].set_mutation(1.0)  # Change the original mutation rate
    assert binary_chromosome.genes[0].get_mutation() != binary_first_half.genes[0].get_mutation()  # Should be different as the original mutation rate changed


# Run the tests
if __name__ == "__main__":
    pytest.main()
