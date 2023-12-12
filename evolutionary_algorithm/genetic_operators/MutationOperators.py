from typing import List

import numpy as np

from evolutionary_algorithm.chromosome.Chromosome import Chromosome


def perform_bit_flip_mutation(random_generator, chromosome: Chromosome) -> None:
    """
    Perform bit flip mutation on the given chromosome.

    Args:
        chromosome (Chromosome): The chromosome on which bit flip mutation will be performed.
    """
    for part_chromosome in chromosome.part_chromosomes:
        _perform_part_chromosome_bit_flip_mutation(random_generator, part_chromosome)


def _perform_part_chromosome_bit_flip_mutation(random_generator, part_chromosome: List[int]) -> None:
    """
    Perform bit flip mutation on a single part chromosome.

    Args:
        part_chromosome (PartChromosome): The part chromosome on which bit flip mutation will be performed.
    """
    for gene in part_chromosome.genes:
        if random_generator.uniform(0, 1) < gene.get_mutation():
            gene_value = gene.get_gene_value()
            mutated_value = 1 - gene_value  # Bit flip mutation: flip 0 to 1 or 1 to 0
            gene.set_gene_value(mutated_value)


def perform_gaussian_mutation(random_generator, chromosome: Chromosome, mu: float, sigma: float) -> None:
    """
    Perform Gaussian mutation on the given chromosome.

    Args:
        chromosome (Chromosome): The chromosome on which Gaussian mutation will be performed.
        sigma (float): The standard deviation of the Gaussian mutation, it's strength.
        mu (float): The mean of the Gaussian distribution, paired with the sigma, these control the shape.
    """
    for part_chromosome in chromosome.part_chromosomes:
        _perform_part_chromosome_gaussian_mutation(random_generator, part_chromosome, mu, sigma)


def _perform_part_chromosome_gaussian_mutation(random_generator, part_chromosome: List[int], mu: float, sigma: float) -> None:
    """
    Perform Gaussian mutation on a single part chromosome.

    Args:
        part_chromosome (PartChromosome): The part chromosome on which Gaussian mutation will be performed.
        sigma (float): The standard deviation of the Gaussian mutation, it's strength.
        mu (float): The mean of the Gaussian distribution, paired with the sigma, these control the shape.
    """
    for gene in part_chromosome.genes:  # For each gene in the part chromosome
        if random_generator.uniform(0, 1) < gene.get_mutation():  # Check if the gene should mutate based on its mutation rate
            gene_value = gene.get_gene_value()  # Grab the current gene value
            mutated_value = gene_value + random_generator.normal(mu, sigma)  # Add noise to get the mutated value

            # Check if the mutated value exceeds the gene's valid range and adjust it if necessary
            if mutated_value > gene.get_gene_max():  # If mutated value is above the gene max, subtract the overflow
                gene.set_gene_value(gene.get_gene_max() - (mutated_value - gene.get_gene_max()))
            elif mutated_value < gene.get_gene_min():  # If mutated value is below the gene min, add the deficit
                gene.set_gene_value(gene.get_gene_min() + (gene.get_gene_min() - mutated_value))
            else:  # Normal case: set the gene value to the mutated value
                gene.set_gene_value(mutated_value)


def perform_gaussian_mutation_image(random_generator, chromosome, mutation_rate, mu: float, sigma: float) -> None:
    """
    Perform Gaussian mutation on the given chromosome.

    Args:
        chromosome (Chromosome): The chromosome on which Gaussian mutation will be performed.
        sigma (float): The standard deviation of the Gaussian mutation, it's strength.
        mu (float): The mean of the Gaussian distribution, paired with the sigma, these control the shape.
    """
    """
    Apply Gaussian mutation to a chromosome.

    Args:
        chromosome (np.array): The chromosome (image) to be mutated.
        mutation_rate (float): Probability of mutating each pixel.
        mean (float): Mean of the Gaussian distribution.
        std_dev (float): Standard deviation of the Gaussian distribution.
    """
    # Iterate over each pixel in the grayscale image
    for i in range(chromosome.shape[0]):
        for j in range(chromosome.shape[1]):
            if random_generator.random() < mutation_rate:
                # Apply Gaussian noise
                noise = random_generator.normal(mu, sigma)
                # Add noise to the pixel value and clip to ensure it remains in valid range
                chromosome[i, j] += noise
                chromosome[i, j] = np.clip(chromosome[i, j], 0, 255)


def perform_gaussian_mutation_dcm_patch(random_generator, chromosome, mutation_rate, mu, sigma) -> None:
    """
    Apply Gaussian mutation to a randomly selected patch of the chromosome (image).

    Args:
        chromosome (np.array): The chromosome (image) to be mutated.
        mutation_rate (float): Probability of mutating each pixel within the patch.
        mu (float): Mean of the Gaussian distribution.
        sigma (float): Standard deviation of the Gaussian distribution.
    """

    # Randomly determine patch size (e.g., 10% to 50% of the image dimensions)
    patch_height = random_generator.randint(int(chromosome.shape[0] * 0.05), int(chromosome.shape[0] * 0.15))
    patch_width = random_generator.randint(int(chromosome.shape[1] * 0.05), int(chromosome.shape[1] * 0.15))

    # Randomly determine the starting coordinates of the patch
    start_i = random_generator.randint(0, chromosome.shape[0] - patch_height)
    start_j = random_generator.randint(0, chromosome.shape[1] - patch_width)

    # Iterate over the patch
    for i in range(start_i, start_i + patch_height):
        for j in range(start_j, start_j + patch_width):
            if random_generator.random() < mutation_rate:
                # Apply Gaussian noise
                noise = random_generator.normal(mu, sigma)
                # Add noise to the pixel value and clip to ensure it remains in valid range
                chromosome[i, j] += noise
                chromosome[i, j] = np.clip(chromosome[i, j], 1000, 3000)


def perform_gaussian_mutation_dcm_image(random_generator, chromosome, mutation_rate, mu: float, sigma: float) -> None:
    """
    Perform Gaussian mutation on the given chromosome.

    Args:
        chromosome (Chromosome): The chromosome on which Gaussian mutation will be performed.
        sigma (float): The standard deviation of the Gaussian mutation, it's strength.
        mu (float): The mean of the Gaussian distribution, paired with the sigma, these control the shape.
    """
    """
    Apply Gaussian mutation to a chromosome.

    Args:
        chromosome (np.array): The chromosome (image) to be mutated.
        mutation_rate (float): Probability of mutating each pixel.
        mean (float): Mean of the Gaussian distribution.
        std_dev (float): Standard deviation of the Gaussian distribution.
    """
    counter = 0
    # Iterate over each pixel in the grayscale image
    for i in range(chromosome.shape[0]):
        for j in range(chromosome.shape[1]):
            if random_generator.random() < mutation_rate:
                # Apply Gaussian noise
                noise = random_generator.normal(mu, sigma)
                # print (f"noise: {noise}")
                # Add noise to the pixel value and clip to ensure it remains in valid range
                chromosome[i, j] += noise
                chromosome[i, j] = np.clip(chromosome[i, j], 0, 4095)
                counter += 1

    # print(f"counter in mututaion: {counter}")
