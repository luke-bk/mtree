import pytest

from evolutionary_algorithm.gene.MtreeGene import MtreeGene


def test_gene_initialization():
    """
    Test if the gene is initialized correctly.
    """
    gene = MtreeGene()  # dominance and mutation values should init between 0 and 1
    assert 0 <= gene.get_dominance() <= 1
    assert 0 <= gene.get_mutation() <= 1


def test_gene_copy():
    """
    Test if the gene can be deep copied and the attribute values are retained.
    """
    gene = MtreeGene()  # original gene
    gene_copy = gene.copy_gene()  # copy of the original (deep copy)
    assert gene.get_gene_value() == gene_copy.get_gene_value()
    assert gene.get_dominance() == gene_copy.get_dominance()
    assert gene.get_mutation() == gene_copy.get_mutation()


def test_gene_copy_changes():
    """
    Test if modifying the attributes of the gene copy doesn't affect the original gene.
    """
    gene = MtreeGene()
    gene_copy = gene.copy_gene()

    # Modify the attributes of the gene copy
    gene_copy.set_dominance(0.3)
    gene_copy.set_mutation(0.8)

    # Assert that the modified attributes of the gene copy do not affect the original gene
    assert gene.get_dominance() != gene_copy.get_dominance()
    assert gene.get_mutation() != gene_copy.get_mutation()


def test_gene_setters():
    """
    Test the setter methods of the gene.
    """
    gene = MtreeGene()
    gene.set_dominance(0.5)
    gene.set_mutation(0.3)

    assert gene.get_dominance() == 0.5
    assert gene.get_mutation() == 0.3


def test_gene_setters_invalid_values():
    """
    Test if the setter methods raise a ValueError for invalid values.
    """
    gene = MtreeGene()  # by default dom and mutation are between 0 and 1
    with pytest.raises(ValueError):
        gene.set_dominance(-0.5)
    with pytest.raises(ValueError):
        gene.set_dominance(1.5)
    with pytest.raises(ValueError):
        gene.set_mutation(-0.2)
    with pytest.raises(ValueError):
        gene.set_mutation(1.2)


def test_gene_boundary_values():
    """
    Test the behavior of the gene class at the boundary values.
    """
    # Test the lower bounds
    gene_lower = MtreeGene()
    gene_lower.set_dominance(0)
    gene_lower.set_mutation(0)
    assert gene_lower.get_dominance() == 0
    assert gene_lower.get_mutation() == 0

    # Test setting values slightly below the lower bounds
    with pytest.raises(ValueError):
        gene_lower.set_dominance(-0.1)
    with pytest.raises(ValueError):
        gene_lower.set_mutation(-0.1)

    # Test the upper bounds
    gene_upper = MtreeGene()
    gene_upper.set_dominance(1)
    gene_upper.set_mutation(1)
    assert gene_upper.get_dominance() == 1
    assert gene_upper.get_mutation() == 1

    # Test setting values slightly above the upper bounds
    with pytest.raises(ValueError):
        gene_upper.set_dominance(1.1)
    with pytest.raises(ValueError):
        gene_upper.set_mutation(1.1)
