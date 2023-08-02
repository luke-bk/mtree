from typing import List
import evolutionary_algorithm.chromosome.Chromosome
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
            generation (int): The generation when the population was created.
            parent_population (Population, optional): The parent population of this population. Defaults to None.
        """
        self.name: str = name  # The name of the population, the root pop will always be 0
        self.generation: int = generation  # The generation the population was created
        self.parent_population: Population = parent_population  # A link to its parent population, if it has one
        self.child_populations: Population = []  # A tuple of populations that are its two children
        self.chromosomes = []  # The individuals in this population, list of chromosomes

    def add_chromosome(self, chromosome):
        """
        Add a chromosome to the population.

        Args:
            chromosome (Chromosome): The chromosome to add.
        """
        self.chromosomes.append(chromosome)

    def split_population(self, generation: int):
        """
        Split the population into two child populations.

        Args:
            generation (int): The current generation the split occured

        Returns:
            Population: The first child population.
            Population: The second child population.
        """
        child1_name = self.name + "0"  # The first child is named after the parent
        child2_name = self.name + "1"  # The second child is also named after the parent
        child1 = Population(child1_name, generation, parent_population=self)  # Create the first child population
        child2 = Population(child2_name, generation, parent_population=self)  # Create the second child population

        # Split chromosomes between the two child populations
        num_chromosomes = len(self.chromosomes)  # The number of individuals in the population
        split_point = num_chromosomes // 2  # calculate the mid point of the population using integer division

        # Handle odd-length population sizes by making the first half one element longer
        child_1_population_size = split_point + (num_chromosomes % 2)  # If there is an odd length, pop 1 gets +1 inds
        child_2_population_size = split_point

        #########################HERE NEEDS SUS WHEN HALVING THE POPUTLATION SIZE#######################################



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
            f"Population: {self.name}, created generation: {self.generation}, parent population: {self.parent_population}")
        for i, chromosome in enumerate(self.chromosomes):
            print(f"Chromosome {i + 1}:")
            chromosome.print_values()
            print()

    def print_population_expressed_form(self):
        """
        Print the expressed form of each chromosome in the population.
        """
        print(
            f"Population: {self.name}, created generation: {self.generation}, parent population: {self.parent_population}")
        for i, chromosome in enumerate(self.chromosomes):
            print(f"Chromosome {i + 1}:")
            print(chromosome.express_highest_dominance())

    def print_population_human_readable(self):
        """
        Print the expressed form of each chromosome in the population.
        """
        print(
            f"Population: {self.name}, created generation: {self.generation}, parent population: {self.parent_population}")
        for i, chromosome in enumerate(self.chromosomes):
            print(f"Chromosome {i + 1}:", end="")
            print(chromosome.express_highest_dominance(), ": ", chromosome.get_fitness())
