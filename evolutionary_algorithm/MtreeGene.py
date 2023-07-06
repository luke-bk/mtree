import numpy as np  # So we can use Mersenne Twister, instead of the regular random library
import copy  # Make use of the deep copy functionality


class MtreeGene:
    """
    A class representing a gene in the M-tree algorithm.

    Attributes:
        name (str): A unique name for this gene based on its hash of memory, this will change every run.
        gene_min (float): The minimum value of the gene.
        gene_max (float): The maximum value of the gene.
        gene_value (float): The value of the gene, between min and max.
        dominance (float): The dominance value of the gene, between 0 and 1.
        mutation (float): The mutation value of the gene, between 0 and 1.
    """

    def __init__(self, gene_min: float, gene_max: float, dominance_min: float = 0.0,
                 dominance_max: float = 1.0, mutation_min: float = 0.0, mutation_max: float = 1.0) -> None:
        """
        Constructor that initializes the MtreeGene instance.

        Args:
            gene_min (float): The minimum value of the gene.
            gene_max (float): The maximum value of the gene.
            dominance_min (float, optional): The minimum value of the dominance. Defaults to 0.0.
            dominance_max (float, optional): The maximum value of the dominance. Defaults to 1.0.
            mutation_min (float, optional): The minimum value of the mutation. Defaults to 0.0.
            mutation_max (float, optional): The maximum value of the mutation. Defaults to 1.0.
        """
        self.set_name()
        self.gene_min: float = gene_min
        self.gene_max: float = gene_max
        self.gene_value: float = np.random.uniform(gene_min, gene_max)

        self.dominance_min: float = dominance_min
        self.dominance_max: float = dominance_max
        self.dominance: float = np.random.uniform(dominance_min, dominance_max)

        self.mutation_min: float = mutation_min
        self.mutation_max: float = mutation_max
        self.mutation: float = np.random.uniform(mutation_min, mutation_max)

    def copy_gene(self) -> 'MtreeGene':
        """
        Returns a deep copy of the MtreeGene instance.

        Returns:
            MtreeGene: A deep copy of the MtreeGene instance.
        """
        copy_instance = copy.deepcopy(self)
        copy_instance.set_name()
        return copy_instance

    # Getter methods
    def get_name(self) -> str:
        return self.name

    def get_gene_min(self) -> float:
        return self.gene_min

    def get_gene_max(self) -> float:
        return self.gene_max

    def get_gene_value(self) -> float:
        return self.gene_value

    def get_dominance(self) -> float:
        return self.dominance

    def get_mutation(self) -> float:
        return self.mutation

    # Setter methods
    def set_name(self) -> None:
        """
        Set the gene name, this is derived from the hex id of the memory location.

        """
        self.name = hex(id(self))

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
            raise ValueError(f"Dominance value must be between {self.gene_min} and {self.gene_max}.")

    def set_dominance(self, dominance: float) -> None:
        """
        Set the dominance value of the gene.

        Args:
            dominance (float): The dominance value to be set.

        Raises:
            ValueError: If the dominance value is outside the range [dominance_min, dominance_max].
        """
        if self.dominance_min <= dominance <= self.dominance_max:
            self.dominance = dominance
        else:
            raise ValueError(f"Dominance value must be between {self.dominance_min} and {self.dominance_max}.")

    def set_mutation(self, mutation: float) -> None:
        """
        Set the mutation value of the gene.

        Args:
            mutation (float): The mutation value to be set.

        Raises:
            ValueError: If the mutation value is outside the range [mutation_min, mutation_max].
        """
        if self.mutation_min <= mutation <= self.mutation_max:
            self.mutation = mutation
        else:
            raise ValueError(f"Mutation value must be between {self.mutation_min} and {self.mutation_max}.")
