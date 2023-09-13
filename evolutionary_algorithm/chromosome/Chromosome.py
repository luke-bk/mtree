import copy
from typing import List, Optional

from evolutionary_algorithm.chromosome.PartChromosome import PartChromosome
from evolutionary_algorithm.gene.MtreeGene import MtreeGene


class Chromosome:
    """
    A class representing a chromosome, consisting of two part chromosomes.

    Attributes:
        name (str): The name of the chromosome.
        part_chromosomes (List[PartChromosome]): The list of part chromosomes in the chromosome.
    """

    def __init__(
            self,
            random_generator,
            parent_name: str,
            part_chromosome_length: int,
            gene_type: str,
            gene_min: Optional[float] = None,
            gene_max: Optional[float] = None
    ) -> None:
        """
        Constructor that initializes the Chromosome instance.

        Args:
            parent_name (str): The name of its parent..if its been split
            part_chromosome_length (int): The length of each part chromosome.
            gene_type (str): The type of MtreeGene ('real' or 'binary').
            gene_min (float, optional): The minimum value for real-valued genes. Defaults to None.
            gene_max (float, optional): The maximum value for real-valued genes. Defaults to None.
        """
        self.name = ""
        self.fitness = None
        self.set_name()  # set the unique name
        self.length = part_chromosome_length
        self.parent_name = parent_name  # Set the parent name provided when the constructor is called
        self.part_chromosomes = [
            PartChromosome(random_generator, self.name, part_chromosome_length, gene_type, gene_min, gene_max),
            PartChromosome(random_generator, self.name, part_chromosome_length, gene_type, gene_min, gene_max)
        ]

    def set_name(self) -> None:
        """
        Set the part chromosomes name, this is derived from the hex id of the memory location.

        """
        self.name = hex(id(self))

    def set_fitness(self, fitness_score: float) -> None:
        """
        Set the fitness of the chromosome.

        """
        self.fitness = fitness_score

    def get_fitness(self) -> None:
        """
        Returns the fitness of the chromosome.

        Returns:
            float self.fitness: A float representation of the fitness score (how good it is at solving the problem)
        """
        return self.fitness

    def split_chromosome(self):
        """
        Split the chromosome into two halves and return deep copies of each half.

        Returns:
            tuple[Chromosome, Chromosome]: A tuple of two Chromosome instances, each containing a deep copy of the
                respective half.
        """
        first_chromosome = self.clone()
        second_chromosome = self.clone()
        part_one_first_half, part_one_second_half = self.part_chromosomes[
            0].split_chromosome()  # Split first part and create clones
        part_two_first_half, part_two_second_half = self.part_chromosomes[
            1].split_chromosome()  # Split second part and create clones

        first_chromosome.part_chromosomes[0] = part_one_first_half
        first_chromosome.part_chromosomes[1] = part_two_first_half

        second_chromosome.part_chromosomes[0] = part_one_second_half
        second_chromosome.part_chromosomes[1] = part_two_second_half

        return first_chromosome, second_chromosome

    def merge_chromosome(self, input_chromosome, child_number: int) -> None:
        """
        Merges this chromosome, with the input into one.

        """
        if child_number == 1:  # It is the right child, so we extend the chromsome
            # Extend the genes in the first part chromosome of the current chromosome with the genes from
            # input_chromosome's first part chromosome
            self.part_chromosomes[0].genes.extend(input_chromosome.part_chromosomes[0].genes)
            # Extend the genes in the second part chromosome of the current chromosome with the genes from
            # input_chromosome's second part chromosome
            self.part_chromosomes[1].genes.extend(input_chromosome.part_chromosomes[1].genes)
        else:  # It is the left child, so we insert at the beginning
            # Extend the genes in the first part chromosome of the current chromosome with the genes from
            # input_chromosome's first part chromosome
            self.part_chromosomes[0].genes.insert(0, input_chromosome.part_chromosomes[0].genes)
            # Extend the genes in the second part chromosome of the current chromosome with the genes from
            # input_chromosome's second part chromosome
            self.part_chromosomes[1].genes.insert(0, input_chromosome.part_chromosomes[1].genes)


    def clone(self) -> 'Chromosome':
        """
        Create a deep copy of the chromosome.

        Returns:
            Chromosome: A deep copy of the chromosome.
        """
        copy_instance = copy.deepcopy(self)  # Deep copy the actual part chromosome
        copy_instance.set_name()  # Rename the chromosome
        copy_instance.part_chromosomes[0].set_name()  # Rename the chromosome
        copy_instance.part_chromosomes[1].set_name()  # Rename the chromosome

        return copy_instance

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

    def get_highest_dominance_genes(self) -> List[MtreeGene]:
        """
        Express the value with the highest dominance for each gene at the same subscript in the two part chromosomes.

        Returns:
            List[MtreeGene]: A list of genes, where each gene is the highest dominance gene.
        """
        expressed_values = []
        for i in range(len(self.part_chromosomes[0].genes)):
            gene1 = self.part_chromosomes[0].get_gene(i)
            gene2 = self.part_chromosomes[1].get_gene(i)

            if gene1.get_dominance() >= gene2.get_dominance():
                expressed_values.append(gene1)
            else:
                expressed_values.append(gene2)

        return expressed_values

    def print_values(self) -> None:
        """
        Print the values of the part chromosomes in the chromosome.
        """
        print(f"Chromosome {self.name}:")
        for i, part_chromosome in enumerate(self.part_chromosomes):
            print(end="    ")
            print(f"Part Chromosome {i + 1} ({part_chromosome.name}):")
            part_chromosome.print_values()

    def print_values_simple(self) -> None:
        """
        Print the values of the part chromosomes in the chromosome.
        """
        for i, part_chromosome in enumerate(self.part_chromosomes):
            print(end="    ")
            part_chromosome.print_values_simple()

    def print_values_verbose(self) -> None:
        """
        Print the values of the part chromosomes in the chromosome.
        """
        print(f"Chromosome {self.name}:")
        for i, part_chromosome in enumerate(self.part_chromosomes):
            print(end="    ")
            print(f"Part Chromosome {i + 1} ({part_chromosome.name}):")
            part_chromosome.print_values_verbose()

    def print_values_expressed(self) -> None:
        """
        Print the values of the expressed chromosomes
        """
        print(" ")
        for genes in enumerate(self.express_highest_dominance()):
            print(f"{genes[1]}, ", end="")
        print(" ")

