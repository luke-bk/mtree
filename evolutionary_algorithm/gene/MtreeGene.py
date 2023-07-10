import numpy as np  # So we can use Mersenne Twister, instead of the regular random library
import copy  # Make use of the deep copy functionality


class MtreeGene:
    """
    A parent class representing a gene in the M-tree algorithm. A child should be a bit-string or a real value number.
    The unique thing about this gene is that it can contain dominance, mutation and other various parameter values, with
    the intention of placing all these EA parameter values inside the gene, rather then having to tune them.

    Attributes:
        name (str): A unique name for this gene based on its hash of memory, this will change every run.
        dominance (float): The dominance value of the gene, between 0 and 1.
        mutation (float): The mutation value of the gene, between 0 and 1.
    """

    def __init__(self, dominance_min: float = 0.0, dominance_max: float = 1.0, mutation_min: float = 0.0,
                 mutation_max: float = 1.0) -> None:
        """
        Constructor that initializes the MtreeGene instance.

        Args:

            dominance_min (float, optional): The minimum value of the dominance. Defaults to 0.0.
            dominance_max (float, optional): The maximum value of the dominance. Defaults to 1.0.
            mutation_min (float, optional): The minimum value of the mutation. Defaults to 0.0.
            mutation_max (float, optional): The maximum value of the mutation. Defaults to 1.0.
        """
        self.name = ""
        self.set_name()
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
