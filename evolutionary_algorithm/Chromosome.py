from evolutionary_algorithm.PartChromosome import PartChromosome


class Chromosome:
    def __init__(self, length: int, gene_min: float, gene_max: float):
        """
        Initialize a Chromosome with two PartChromosomes.

        :param length: The length of each PartChromosome (number of genes).
        :param gene_min: The minimum value for each gene.
        :param gene_max: The maximum value for each gene.
        """
        self.set_name()
        self.part_chromosomes = [PartChromosome(length, gene_min, gene_max) for _ in range(2)]
        self.length = length

    def deep_clone(self) -> 'Chromosome':
        """
        Create a deep clone of the chromosome.

        :return: A deep clone of the chromosome.
        """
        clone = Chromosome(self.name, self.length, 0.0, 0.0)  # Create a new Chromosome object with the same name and length
        clone.part_chromosomes = [part_chromosome.deep_clone() for part_chromosome in self.part_chromosomes]
        clone.set_name()
        return clone

    def print_values(self) -> None:
        """
        Print the values of both part chromosomes.
        """
        print(f"Chromosome {self.name}")
        for i, part_chromosome in enumerate(self.part_chromosomes):
            print(f"Part Chromosome {i + 1} values:")
            for j, gene in enumerate(part_chromosome.chromosome):
                print(f"Gene {j + 1}: {gene.get_gene_value()}")

    def print_values_verbose(self) -> None:
        """
        Print the values of both part chromosomes with the dominance values, and mutation values.
        """
        print(f"Chromosome {self.name}")
        for i, part_chromosome in enumerate(self.part_chromosomes):
            print(f"Part Chromosome {i + 1} values:")
            for j, gene in enumerate(part_chromosome.chromosome):
                print(f"Gene {j + 1}: {gene.get_gene_value()}, dominance:  {gene.get_dominance()}, mutation: "
                      f"{gene.get_mutation()}")

    def get_chromosome_length(self) -> int:
        """
        Get the length of the chromosome (number of genes).

        :return: The length of the chromosome.
        """
        return self.length

    def set_name(self) -> None:
        """
        Set the chromosomes name, this is derived from the hex id of the memory location.

        """
        self.name = hex(id(self))
