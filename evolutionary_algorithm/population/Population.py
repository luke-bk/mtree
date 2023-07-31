from typing import List
import copy
import numpy as np


class Population:
    """
    A class representing a population of Chromosomes.

    Attributes:
        name (str): The name of the population.
        generation (int): The generation number of the population.
        parent_population (Population): The parent population of this population.
        chromosomes (List[Chromosome]): The list of chromosomes in the population.
    """

    def __init__(self, name: str, generation: int, parent_population=None):
        """
        Constructor that initializes the Population instance.

        Args:
            name (str): The name of the population.
            generation (int): The generation number of the population.
            parent_population (Population, optional): The parent population of this population. Defaults to None.
        """
        self.name = name
        self.generation = generation
        self.parent_population = parent_population
        self.chromosomes = []

    def add_chromosome(self, chromosome):
        """
        Add a chromosome to the population.

        Args:
            chromosome (Chromosome): The chromosome to add.
        """
        self.chromosomes.append(chromosome)

    def split_population(self, child1_name, child2_name):
        """
        Split the population into two child populations.

        Args:
            child1_name (str): The name of the first child population.
            child2_name (str): The name of the second child population.

        Returns:
            Population: The first child population.
            Population: The second child population.
        """
        child1 = Population(child1_name, self.generation + 1, parent_population=self)
        child2 = Population(child2_name, self.generation + 1, parent_population=self)

        # Split chromosomes between the two child populations
        num_chromosomes = len(self.chromosomes)
        split_point = num_chromosomes // 2

        for i in range(split_point):
            child1.add_chromosome(self.chromosomes[i].split_chromosome())

        for i in range(split_point, num_chromosomes):
            child2.add_chromosome(self.chromosomes[i].split_chromosome())

        return child1, child2

    def merge_populations(self, child1, child2):
        """
        Merge two child populations back into the parent population.

        Args:
            child1 (Population): The first child population.
            child2 (Population): The second child population.
        """
        # Ensure that the input child populations have the same parent
        if child1.parent_population != child2.parent_population:
            raise ValueError("Both child populations must have the same parent population.")

        # Clear the current chromosomes in the parent population
        self.chromosomes.clear()

        # Merge chromosomes from child1 and child2 to form new chromosomes for the parent population
        num_chromosomes = len(child1.chromosomes)
        for i in range(num_chromosomes):
            new_chromosome = child1.chromosomes[i].clone()
            new_chromosome.merge_chromosome(child1.chromosomes[i], child2.chromosomes[i])
            self.add_chromosome(new_chromosome)

    def get_name(self):
        """
        Get the name of the population.

        Returns:
            str: The name of the population.
        """
        return self.name

    def get_generation(self):
        """
        Get the generation number of the population.

        Returns:
            int: The generation number of the population.
        """
        return self.generation

    def get_parent_population(self):
        """
        Get the parent population of this population.

        Returns:
            Population: The parent population of this population.
        """
        return self.parent_population

    def print_population(self):
        """
        Print the details of each chromosome in the population.
        """
        print(
            f"Population: {self.name}, Generation: {self.generation}, Parent Population: {self.parent_population}")
        for i, chromosome in enumerate(self.chromosomes):
            print(f"Chromosome {i + 1}:")
            chromosome.print_values()
            print()
