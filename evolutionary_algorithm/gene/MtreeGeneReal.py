from evolutionary_algorithm.gene.MtreeGene import MtreeGene


class MtreeGeneReal(MtreeGene):
    """
    A subclass representing a real-valued gene in the M-tree algorithm.
    Inherits from the MtreeGene class.

    Attributes:
        name (str): A unique name for this gene based on its hash of memory, this will change every run.
        gene_min (float): The minimum value of the gene.
        gene_max (float): The maximum value of the gene.
        gene_value (float): The value of the gene, between min and max.
        dominance (float): The dominance value of the gene, between 0 and 1.
        mutation (float): The mutation value of the gene, between 0 and 1.
    """

    def __init__(self,random_generator, gene_min: float = 0.0, gene_max: float = 1.0, dominance_min: float = 0.0,
                 dominance_max: float = 1.0, mutation_min: float = 0.0, mutation_max: float = 1.0) -> None:
        """
        Constructor that initializes the MtreeGeneReal instance.

        Args:
            gene_min (float): The minimum value of the gene.
            gene_max (float): The maximum value of the gene.
            dominance_min (float, optional): The minimum value of the dominance. Defaults to 0.0.
            dominance_max (float, optional): The maximum value of the dominance. Defaults to 1.0.
            mutation_min (float, optional): The minimum value of the mutation. Defaults to 0.0.
            mutation_max (float, optional): The maximum value of the mutation. Defaults to 1.0.
        """
        super().__init__(dominance_min, dominance_max, mutation_min, mutation_max)
        self.gene_min: float = gene_min
        self.gene_max: float = gene_max
        self.gene_value: float = random_generator.uniform(gene_min, gene_max)

    # Additional methods specific to MtreeGeneReal can be added here
    # Getter methods
    def get_gene_min(self) -> float:
        return self.gene_min

    def get_gene_max(self) -> float:
        return self.gene_max

    def get_gene_value(self) -> float:
        return self.gene_value

    # Setter methods
    def set_gene_min(self, gene_min: float) -> None:
        self.gene_min = gene_min

    def set_gene_max(self, gene_max: float) -> None:
        self.gene_max = gene_max

    def set_gene_value(self, gene_value: float) -> None:
        """
        Set the value value of the gene.

        Args:
            gene_value (float): The gene value to be set.

        Raises:
            ValueError: If the gene value is outside the range [gene_value_min, gene_value_max].
        """
        if self.gene_min <= gene_value <= self.gene_max:
            self.gene_value = gene_value
        else:
            raise ValueError(f"Gene value must be between {self.gene_min} and {self.gene_max}.")