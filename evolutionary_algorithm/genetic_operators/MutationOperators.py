import numpy as np
from typing import List

from evolutionary_algorithm.chromosome.Chromosome import Chromosome


class MutationOperators:
    """
    A static class representing mutation operators for performing bit flip and Gaussian mutation on a chromosome.
    """

    @staticmethod
    def perform_bit_flip_mutation(chromosome: Chromosome) -> None:
        """
        Perform bit flip mutation on the given chromosome.

        Args:
            chromosome (Chromosome): The chromosome on which bit flip mutation will be performed.
        """
        for part_chromosome in chromosome.part_chromosomes:
            MutationOperators._perform_part_chromosome_bit_flip_mutation(part_chromosome)

    @staticmethod
    def _perform_part_chromosome_bit_flip_mutation(part_chromosome: List[int]) -> None:
        """
        Perform bit flip mutation on a single part chromosome.

        Args:
            part_chromosome (PartChromosome): The part chromosome on which bit flip mutation will be performed.
        """
        for gene in part_chromosome.genes:
            if np.random.rand() < gene.get_mutation():
                gene_value = gene.get_gene_value()
                mutated_value = 1 - gene_value  # Bit flip mutation: flip 0 to 1 or 1 to 0
                gene.set_gene_value(mutated_value)

    @staticmethod
    def perform_gaussian_mutation(chromosome: Chromosome, mu: float, sigma: float) -> None:
        """
        Perform Gaussian mutation on the given chromosome.

        Args:
            chromosome (Chromosome): The chromosome on which Gaussian mutation will be performed.
            sigma (float): The standard deviation of the Gaussian mutation, it's strength.
            mu (float): The mean of the Gaussian distribution, paired with the sigma, these control the shape.
        """
        for part_chromosome in chromosome.part_chromosomes:
            MutationOperators._perform_part_chromosome_gaussian_mutation(part_chromosome, mu, sigma)

    @staticmethod
    def _perform_part_chromosome_gaussian_mutation(part_chromosome: List[int], mu: float, sigma: float) -> None:
        """
        Perform Gaussian mutation on a single part chromosome.

        Args:
            part_chromosome (PartChromosome): The part chromosome on which Gaussian mutation will be performed.
            sigma (float): The standard deviation of the Gaussian mutation, it's strength.
            mu (float): The mean of the Gaussian distribution, paired with the sigma, these control the shape.
        """
        for gene in part_chromosome.genes:  # For each gene in the part chromosome
            if np.random.rand() < gene.get_mutation():  # Check if the gene should mutate
                gene_value = gene.get_gene_value()  # Grab the current gene value
                mutated_value = gene_value + np.random.normal(mu, sigma)  # Add noise to get mutated value
                if mutated_value > gene.get_gene_max():  # If mutated value is above the gene max, subtract leftovers
                    gene.set_gene_value(gene.get_gene_max() - mutated_value)
                elif mutated_value < gene.get_gene_min():  # If mutated value is lower the gene min, add leftovers
                    gene.set_gene_value(gene.get_gene_min() - mutated_value)
                else:  # Normal case
                    gene.set_gene_value(mutated_value)
