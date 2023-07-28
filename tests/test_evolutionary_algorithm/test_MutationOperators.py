import numpy as np
import pytest

from evolutionary_algorithm.chromosome.Chromosome import Chromosome
from evolutionary_algorithm.chromosome.PartChromosome import PartChromosome
from evolutionary_algorithm.gene.MtreeGene import MtreeGene
from evolutionary_algorithm.gene.MtreeGeneBit import MtreeGeneBit
from evolutionary_algorithm.gene.MtreeGeneReal import MtreeGeneReal
from evolutionary_algorithm.genetic_operators.MutationOperators import MutationOperators


class TestMutationOperators:

    # Test case for bit flip mutation
    def test_bit_flip_mutation(self):
        # Create a chromosome with two part chromosomes
        part_chromosomes = [
            PartChromosome("parent1", 10, "bit"),
            PartChromosome("parent2", 10, "bit")
        ]
        chromosome = Chromosome("chromosome", "parent", 10, "bit")
        chromosome.part_chromosomes = part_chromosomes

        # Set gene values and mutation rates for the first part chromosome
        gene1 = MtreeGeneBit()
        gene1.set_gene_value(0)  # It's original value is 0
        gene1.set_mutation(0.0)  # It's mutation rate is 0, therefore it should NOT change with bit flip
        part_chromosomes[0].set_gene(0, gene1)

        # Set gene values and mutation rates for the second part chromosome
        gene2 = MtreeGeneBit()
        gene2.set_gene_value(0)  # It's original value is 0
        gene2.set_mutation(1.0)  # It's mutation rate is 1.0, therefore it SHOULD change with bit flip
        part_chromosomes[1].set_gene(0, gene2)

        # Set gene values and mutation rates for the second part chromosome
        gene3 = MtreeGeneBit()
        gene3.set_gene_value(1)  # It's original value is 1
        gene3.set_mutation(1.0)  # It's mutation rate is 1.0, therefore it SHOULD change with bit flip
        part_chromosomes[1].set_gene(1, gene3)

        # Perform bit flip mutation
        MutationOperators.perform_bit_flip_mutation(chromosome)

        # Verify that gene values were flipped based on the mutation rates
        assert gene1.get_gene_value() == 0  # Gene 1 should NOT have changed from 0 with its mutation rate of 0.0
        assert gene2.get_gene_value() == 1  # Gene 2 SHOULD have changed with its mutation rate of 1.0
        assert gene3.get_gene_value() == 0  # Gene 3 SHOULD have changed with its mutation rate of 1.0

    # Test case for Gaussian mutation
    def test_gaussian_mutation(self):
        # setting mu and sigma for the Gaussian mutation
        mu = 0
        sigma = 0.1

        # Create a chromosome with two part chromosomes
        part_chromosomes = [
            PartChromosome("parent1", 10, "real", -1.0, 1.0),
            PartChromosome("parent2", 10, "real", -1.0, 1.0)
        ]
        chromosome = Chromosome("chromosome", "parent", 10, "real", -1.0, 1.0)
        chromosome.part_chromosomes = part_chromosomes

        # Set gene values and mutation rates for the first part chromosome
        gene1 = MtreeGeneReal(-1.0, 1.0)  # Gene between -1 and 1
        gene1.set_gene_value(0.5)  # Set the gene value to 0.5
        gene1.set_mutation(0.0)  # Set the mutation rate to 0, we don't want this gene to mutate
        part_chromosomes[0].set_gene(0, gene1)

        # Set gene values and mutation rates for the second part chromosome
        gene2 = MtreeGeneReal(-1.0, 1.0)  # Gene between -1 and 1
        gene2.set_gene_value(0.5)   # Set the gene value to 0.5
        gene2.set_mutation(1.0)  # Set the mutation rate to 1.0, we want this gene to mutate
        part_chromosomes[1].set_gene(0, gene2)

        # Perform Gaussian mutation
        MutationOperators.perform_gaussian_mutation(chromosome, mu, sigma)

        # Verify that gene values were updated based on the Gaussian distribution
        assert gene1.get_gene_value() == 0.5  # Gene 1 should NOT have changed from 0 with its mutation rate of 0.0
        assert gene2.get_gene_value() != 0.5  # Gene 2 SHOULD have changed with its mutation rate of 1.0
