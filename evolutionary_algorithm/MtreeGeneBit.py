import numpy as np  # So we can use Mersenne Twister, instead of the regular random library

from evolutionary_algorithm.MtreeGene import MtreeGene

# Constants in our bit string... either 0 or 1
ZERO: int = 0
ONE: int = 1


class MtreeGeneBit(MtreeGene):
    """
    A subclass representing a bit-valued gene in the M-tree algorithm.
    Inherits from the MtreeGene class and gene values are either [0,1].

    Attributes:
        name (str): A unique name for this gene based on its hash of memory, this will change every run.
        gene_value (int): The value of the gene, between min and max.
        dominance (float): The dominance value of the gene, between 0 and 1.
        mutation (float): The mutation value of the gene, between 0 and 1.
    """

    def __init__(self, dominance_min: float = 0.0,
                 dominance_max: float = 1.0, mutation_min: float = 0.0, mutation_max: float = 1.0) -> None:
        """
        Constructor that initializes the MtreeGeneReal instance.

        Args:
            dominance_min (float, optional): The minimum value of the dominance. Defaults to 0.0.
            dominance_max (float, optional): The maximum value of the dominance. Defaults to 1.0.
            mutation_min (float, optional): The minimum value of the mutation. Defaults to 0.0.
            mutation_max (float, optional): The maximum value of the mutation. Defaults to 1.0.
        """
        super().__init__(dominance_min, dominance_max, mutation_min, mutation_max)
        self.gene_value: int = np.random.choice([0, 1])

    # Additional methods specific to MtreeGeneReal can be added here
    # Getter methods
    def get_gene_min(self) -> int:
        return ZERO

    def get_gene_max(self) -> int:
        return ONE

    def get_gene_value(self) -> int:
        return self.gene_value

    # Setter methods
    # Setter method
    def set_gene_value(self, gene_value: int) -> None:
        """
        Set the value of the gene.

        Args:
            gene_value (int): The gene value to be set.

        Raises:
            ValueError: If the gene value is not 0 or 1.
        """
        if gene_value in [0, 1]:
            self.gene_value = gene_value
        else:
            raise ValueError("Gene value must be either 0 or 1.")
