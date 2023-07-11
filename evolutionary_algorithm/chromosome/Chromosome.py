import copy
from typing import List, Optional

from evolutionary_algorithm.chromosome.PartChromosome import PartChromosome


class Chromosome:
    """
    A class representing a chromosome, consisting of two part chromosomes.

    Attributes:
        name (str): The name of the chromosome.
        part_chromosomes (List[PartChromosome]): The list of part chromosomes in the chromosome.
    """

    def __init__(
            self,
            name: str,
            part_chromosome_length: int,
            gene_type: str,
            gene_min: Optional[float] = None,
            gene_max: Optional[float] = None
    ) -> None:
        """
        Constructor that initializes the Chromosome instance.

        Args:
            name (str): The name of the chromosome.
            part_chromosome_length (int): The length of each part chromosome.
            gene_type (str): The type of MtreeGene ('real' or 'binary').
            gene_min (float, optional): The minimum value for real-valued genes. Defaults to None.
            gene_max (float, optional): The maximum value for real-valued genes. Defaults to None.
        """
        self.name = name
        self.part_chromosomes = [
            PartChromosome(name, part_chromosome_length, gene_type, gene_min, gene_max),
            PartChromosome(name, part_chromosome_length, gene_type, gene_min, gene_max)
        ]

    def split_chromosome(self) -> tuple['Chromosome', 'Chromosome']:
        """
        Split the chromosome into two halves and return deep copies of each half.

        Returns:
            tuple[Chromosome, Chromosome]: A tuple of two Chromosome instances, each containing a deep copy of the
                respective half.
        """
        first_half, second_half = self.part_chromosomes[0].split_chromosome()
        first_chromosome = Chromosome(self.name, 0, "")
        first_chromosome.part_chromosomes[0] = first_half

        second_chromosome = Chromosome(self.name, 0, "")
        second_chromosome.part_chromosomes[0] = second_half

        return first_chromosome, second_chromosome

    def clone(self) -> 'Chromosome':
        """
        Create a deep copy of the chromosome.

        Returns:
            Chromosome: A deep copy of the chromosome.
        """
        cloned_chromosome = Chromosome(self.name, 0, "")
        cloned_chromosome.part_chromosomes = [
            part_chromosome._deep_copy_part_chromosome(part_chromosome.genes)
            for part_chromosome in self.part_chromosomes
        ]
        return cloned_chromosome

    def express_highest_dominance(self) -> List[int]:
        """
        Express the value with the highest dominance for each gene at the same subscript in the two part chromosomes.

        Returns:
            List[int]: A list of gene values, where each gene value is expressed with the highest dominance.
        """
        expressed_values = []
        for i in range(len(self.part_chromosomes[0].genes)):
            gene1 = self.part_chromosomes[0].get_gene(i)
            gene2 = self.part_chromosomes[1].get_gene(i)

            if gene1.get_dominance() >= gene2.get_dominance():
                expressed_values.append(gene1.get_gene_value())
            else:
                expressed_values.append(gene2.get_gene_value())

        return expressed_values
