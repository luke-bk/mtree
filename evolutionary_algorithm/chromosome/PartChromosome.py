import copy
from typing import List, Tuple, Optional

from evolutionary_algorithm.gene.MtreeGene import MtreeGene
from evolutionary_algorithm.gene.MtreeGeneBit import MtreeGeneBit
from evolutionary_algorithm.gene.MtreeGeneReal import MtreeGeneReal


class PartChromosome:
    """
    A class representing one part of a two part chromosome. Each full chromosome has two sets of genes, only one
    is expressed.

    Attributes:
        parent_name (str): The name of the whole chromosome.
        genes (List[MtreeGene]): A list of MtreeGenes.
    """

    def __init__(
            self,
            parent_name: str,
            part_chromosome_length: int,
            gene_type: str,
            gene_min: Optional[float] = None,
            gene_max: Optional[float] = None
    ) -> None:
        """
        Constructor that initializes the PartChromosome instance.

        Args:
            parent_name (str): The name of the parent (whole chromosome).
            part_chromosome_length (int, optional): The length of the gene list.
            gene_type (str): The type of MtreeGene ('real' or 'binary'). Defaults to 'real'.
            gene_min (float, optional): The minimum value for real-valued genes. Defaults to None.
            gene_max (float, optional): The maximum value for real-valued genes. Defaults to None.
        """
        self.name = ""  # Set up the name variable on creation
        self.set_name()  # Set the unique name variable based on the hash of the memory location
        self.parent_name = parent_name  # Set the parent name provided when the constructor is called
        self.genes = self._create_genes(part_chromosome_length, gene_type, gene_min, gene_max)  # Create the list of
        # genes based on its length, type and optional minimum and maximum gene values if real's are used

    def _create_genes(self, gene_length: int, gene_type: str, gene_min: Optional[float], gene_max: Optional[float]) -> \
            List[MtreeGene]:
        """
        Create a list of MtreeGenes based on the given length, type, and optional min/max values.

        Args:
            gene_length (int): The length of the gene list.
            gene_type (str): The type of MtreeGene ('real' or 'binary').
            gene_min (float, optional): The minimum value for real-valued genes. Defaults to None.
            gene_max (float, optional): The maximum value for real-valued genes. Defaults to None.

        Returns:
            List[MtreeGene]: A list of MtreeGenes.
        """
        genes = []
        for _ in range(gene_length):
            if gene_type == 'real':
                genes.append(MtreeGeneReal(gene_min, gene_max))
            elif gene_type == 'bit':
                genes.append(MtreeGeneBit())
            else:
                raise ValueError("Invalid gene type. Must be 'real' or 'bit'.")
        return genes

    def get_chromosome_length(self) -> int:
        """
        Get the length of the chromosome (number of genes).

        :return: The length of the chromosome.
        """
        return len(self.genes)

    def get_gene(self, index: int) -> MtreeGene:
        """
        Get the MtreeGene at the specified index.

        :param index: The index of the gene to retrieve.
        :return: The MtreeGene at the specified index.
        """
        return self.genes[index]

    def set_gene(self, index: int, gene: MtreeGene) -> None:
        """
        Set the MtreeGene at the specified index.

        :param index: The index at which to set the gene.
        :param gene: The MtreeGene to set.
        """
        self.genes[index] = gene

    def split_chromosome(self):
        """
        Split the chromosome into two halves and return deep copies of each half.

        :return: A tuple of two lists, each containing a deep copy of the chromosome's halves.
        """
        length = len(self.genes)  # Get the length of the chromosome
        midpoint = length // 2  # Calculate the midpoint

        # Handle odd-length chromosomes by making the first half one element longer
        first_half_length = midpoint + (length % 2)

        first_half = self.genes[:first_half_length]  # Get the first half of the chromosome
        second_half = self.genes[first_half_length:]  # Get the second half of the chromosome

        return self._deep_copy_part_chromosome(first_half), \
               self._deep_copy_part_chromosome(second_half)  # Return deep copies of the halves

    def _deep_copy_part_chromosome(self, chromosome: List[MtreeGene]) -> 'PartChromosome':
        """
        Create a deep copy of the given chromosome.

        :param chromosome: The chromosome to create a deep copy of.
        :return: A deep copy of the chromosome.
        """
        copy_genes = []  # Empty list to deep copy the genes
        for gene in chromosome:  # Iterate over the genes and deep copy them
            copy_genes.append(gene.copy_gene())

        copy_instance = copy.deepcopy(self)  # Deep copy the actual part chromosome
        copy_instance.set_name()  # Rename the part chromosome
        copy_instance.genes = copy_genes  # Assign the deep copy list of genes
        return copy_instance

    def set_name(self) -> None:
        """
        Set the part chromosomes name, this is derived from the hex id of the memory location.

        """
        self.name = hex(id(self))

    def print_values(self) -> None:
        """
        Print the values of the part chromosomes.
        """
        for i, gene in enumerate(self.genes):
            print(end="        ")
            print(f"Gene {i}: {gene.get_gene_value()}")

    def print_values_verbose(self) -> None:
        """
        Print the values of the part chromosomes with the dominance values, and mutation values.
        """
        for i, gene in enumerate(self.genes):
            print(end="        ")
            print(
                f"Gene {i}: : {gene.get_gene_value()}, dominance:  {gene.get_dominance()}, mutation: {gene.get_mutation()}")
