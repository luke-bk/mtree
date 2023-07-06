from typing import List, Tuple

from evolutionary_algorithm.MtreeGene import MtreeGene


class PartChromosome:
    def __init__(self, length: int, gene_min: float, gene_max: float):
        """
        Initialize a PartChromosome with a list of MtreeGene objects.

        :param length: The length of the chromosome (number of genes).
        :param gene_min: The minimum value for each gene.
        :param gene_max: The maximum value for each gene.
        """
        self.chromosome = self._initialize_chromosome(length, gene_min, gene_max)

    def _initialize_chromosome(self, length: int, gene_min: float, gene_max: float) -> List[MtreeGene]:
        """
        Initialize the chromosome with a list of MtreeGene objects.

        :param length: The length of the chromosome (number of genes).
        :param gene_min: The minimum value for each gene.
        :param gene_max: The maximum value for each gene.
        :return: The initialized chromosome as a list of MtreeGene objects.
        """
        chromosome = []
        for _ in range(length):
            gene = MtreeGene(gene_min, gene_max)
            chromosome.append(gene)
        return chromosome

    def get_chromosome_length(self) -> int:
        """
        Get the length of the chromosome (number of genes).

        :return: The length of the chromosome.
        """
        return len(self.chromosome)

    def get_gene(self, index: int) -> MtreeGene:
        """
        Get the MtreeGene at the specified index.

        :param index: The index of the gene to retrieve.
        :return: The MtreeGene at the specified index.
        """
        return self.chromosome[index]

    def set_gene(self, index: int, gene: MtreeGene) -> None:
        """
        Set the MtreeGene at the specified index.

        :param index: The index at which to set the gene.
        :param gene: The MtreeGene to set.
        """
        self.chromosome[index] = gene

    def split_chromosome(self) -> Tuple[List[MtreeGene], List[MtreeGene]]:
        """
        Split the chromosome into two halves and return deep copies of each half.

        :return: A tuple of two lists, each containing a deep copy of the chromosome's halves.
        """
        length = len(self.chromosome)  # Get the length of the chromosome
        midpoint = length // 2  # Calculate the midpoint

        # Handle odd-length chromosomes by making the first half one element longer
        first_half_length = midpoint + (length % 2)

        first_half = self.chromosome[:first_half_length]  # Get the first half of the chromosome
        second_half = self.chromosome[first_half_length:]  # Get the second half of the chromosome

        return self._deep_copy_chromosome(first_half), self._deep_copy_chromosome(
            second_half)  # Return deep copies of the halves

    def _deep_copy_chromosome(self, chromosome: List[MtreeGene]) -> List[MtreeGene]:
        """
        Create a deep copy of the given chromosome.

        :param chromosome: The chromosome to create a deep copy of.
        :return: A deep copy of the chromosome.
        """
        return [gene.copy_gene() for gene in chromosome]
