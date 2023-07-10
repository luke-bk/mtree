import pytest
from evolutionary_algorithm.MtreeGeneReal import MtreeGeneReal


# Test initialization and getter methods
def test_gene_initialization_and_getters():
    gene = MtreeGeneReal(0, 10, 0, 1, 0, 1)
    assert gene.get_gene_min() == 0
    assert gene.get_gene_max() == 10
    assert 0 <= gene.get_gene_value() <= 10


# Test setter methods
def test_gene_setters():
    gene = MtreeGeneReal(0, 10, 0, 1, 0, 1)
    gene.set_gene_min(-5)
    gene.set_gene_max(15)
    gene.set_gene_value(7)
    assert gene.get_gene_min() == -5
    assert gene.get_gene_max() == 15
    assert gene.get_gene_value() == 7


# Test setting invalid gene value
def test_invalid_gene_value():
    gene = MtreeGeneReal(0, 10, 0, 1, 0, 1)
    with pytest.raises(ValueError):
        gene.set_gene_value(-1)  # Below the minimum
    with pytest.raises(ValueError):
        gene.set_gene_value(11)  # Above the maximum
    with pytest.raises(ValueError):
        gene.set_dominance(-1)  # Below the minimum
    with pytest.raises(ValueError):
        gene.set_dominance(11)  # Above the maximum
    with pytest.raises(ValueError):
        gene.set_mutation(-1)  # Below the minimum
    with pytest.raises(ValueError):
        gene.set_mutation(11)  # Above the maximum


# Test setting invalid gene value
def test_invalid_mutation_value():
    gene = MtreeGeneReal(0, 10, 0, 1, 0, 1)
    with pytest.raises(ValueError):
        gene.set_mutation(-1)  # Below the minimum
    with pytest.raises(ValueError):
        gene.set_mutation(11)  # Above the maximum


# Test setting invalid gene value
def test_invalid_dominance_value():
    gene = MtreeGeneReal(0, 10, 0, 1, 0, 1)
    with pytest.raises(ValueError):
        gene.set_dominance(-1)  # Below the minimum
    with pytest.raises(ValueError):
        gene.set_dominance(11)  # Above the maximum


# Test gene copying
def test_gene_copy():
    gene = MtreeGeneReal(0, 10, 0, 1, 0, 1)
    gene_copy = gene.copy_gene()  # Copy the gene
    assert gene.get_gene_min() == gene_copy.get_gene_min()  # We expect the same values
    assert gene.get_gene_max() == gene_copy.get_gene_max()  # We expect the same values
    assert gene.get_gene_value() == gene_copy.get_gene_value()   # We expect the same values
    assert gene.get_name() != gene_copy.get_name()   # We expect a different name


# Test deep copy
def test_deep_copy():
    gene = MtreeGeneReal(0, 10, 0, 1, 0, 1)
    gene_copy = gene.copy_gene()

    # Modify the original gene
    gene.set_gene_value(5)
    gene.set_dominance(0.7)
    gene.set_mutation(0.8)

    # Verify that the copy remains unaffected
    assert gene_copy.get_gene_value() != gene.get_gene_value()  # We expect different values
    assert gene_copy.get_dominance() != gene.get_dominance()  # We expect different values
    assert gene_copy.get_mutation() != gene.get_mutation()  # We expect different values
    assert gene.get_name() != gene_copy.get_name()  # We expect different values


# Test deep copy uniqueness
def test_deep_copy_uniqueness():
    gene1 = MtreeGeneReal(0, 10, 0, 1, 0, 1)
    gene2 = gene1.copy_gene()

    # Modify gene2
    gene2.set_gene_value(7)

    # Verify that gene1 and gene2 are independent
    assert gene1.get_gene_value() != gene2.get_gene_value()


# Test gene name uniqueness
def test_gene_name_uniqueness():
    gene1 = MtreeGeneReal(0, 10, 0, 1, 0, 1)
    gene2 = MtreeGeneReal(0, 10, 0, 1, 0, 1)
    assert gene1.get_name() != gene2.get_name()


# Test boundary values
def test_boundary_values():
    gene = MtreeGeneReal(0, 10, 0, 1, 0, 1)
    # Gene Value lower valid boundary
    gene.set_gene_value(0)
    assert gene.get_gene_value() == 0
    gene.set_gene_value(0.01)
    assert gene.get_gene_value() == 0.01
    # Gene Value upper valid boundary
    gene.set_gene_value(9.99)
    assert gene.get_gene_value() == 9.99
    gene.set_gene_value(10)
    assert gene.get_gene_value() == 10

    # Mutation Value lower valid boundary
    gene.set_mutation(0)
    assert gene.get_mutation() == 0
    gene.set_mutation(0.01)
    assert gene.get_mutation() == 0.01
    # Mutation Value upper valid boundary
    gene.set_mutation(0.99)
    assert gene.get_mutation() == 0.99
    gene.set_mutation(1)
    assert gene.get_mutation() == 1

    # Dominance Value lower valid boundary
    gene.set_dominance(0)
    assert gene.get_dominance() == 0
    gene.set_dominance(0.01)
    assert gene.get_dominance() == 0.01
    # Dominance Value upper valid boundary
    gene.set_dominance(0.99)
    assert gene.get_dominance() == 0.99
    gene.set_dominance(1)
    assert gene.get_dominance() == 1


# Test dominance and mutation range
def test_dominance_and_mutation_range():
    gene = MtreeGeneReal(0, 10, 0, 1, 0, 1)
    assert 0 <= gene.get_dominance() <= 1
    assert 0 <= gene.get_mutation() <= 1


# Test setting dominance and mutation values
def test_dominance_and_mutation_setters():
    gene = MtreeGeneReal(0, 10, 0, 1, 0, 1)
    gene.set_dominance(0.5)
    assert gene.get_dominance() == 0.5
    gene.set_mutation(0.3)
    assert gene.get_mutation() == 0.3


# Test setting invalid dominance and mutation values
def test_invalid_dominance_and_mutation_values():
    gene = MtreeGeneReal(0, 10, 0, 1, 0, 1)
    with pytest.raises(ValueError):
        gene.set_dominance(-0.1)  # Below the minimum
    with pytest.raises(ValueError):
        gene.set_dominance(1.1)  # Above the maximum
    with pytest.raises(ValueError):
        gene.set_mutation(-0.1)  # Below the minimum
    with pytest.raises(ValueError):
        gene.set_mutation(1.1)  # Above the maximum


# Run the tests
if __name__ == '__main__':
    pytest.main()
