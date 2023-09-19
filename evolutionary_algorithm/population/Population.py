from typing import List
import evolutionary_algorithm.chromosome.Chromosome
import copy
import numpy as np

from evolutionary_algorithm.genetic_operators import SelectionOperators


class Population:
    """
    A class representing a population of Chromosomes.

    Attributes:
        name (str): The name of the population.
        generation (int): The generation number of the population.
        parent_population (Population): The parent population of this population.
        chromosomes (List[Chromosome]): The list of chromosomes in the population.
    """

    def __init__(self, random_generator, name: str, generation: int, fitness: int, parent_population=None):
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
        self.best_fitness_at_creation = fitness
        self.random_generator = random_generator
        self.elite = None  # Best chromosome so far
        self.has_improved = False  # Has the sub population achieved a score higher than the best fit at creation?
        self.merge_tracker = 0  # How many generations have passed without surpassing the best fitness at creation.

    def add_chromosome(self, chromosome):
        """
        Add a chromosome to the population.

        Args:
            chromosome (Chromosome): The chromosome to add.
        """
        self.chromosomes.append(chromosome)

    def split_population(self, generation: int, fitness: int):
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
        child_1 = Population(self.random_generator, child1_name, generation, self.elite.get_fitness(),
                             parent_population=self)  # Create the first child population
        child_2 = Population(self.random_generator, child2_name, generation, self.elite.get_fitness(),
                             parent_population=self)  # Create the second child population

        # Get elite
        # Save best current chromosome
        elite = self.get_chromosome_with_max_fitness()

        # Split chromosomes between the two child populations
        num_chromosomes = len(self.chromosomes)  # The number of individuals in the population
        number_of_children = num_chromosomes // 2  # calculate the number of children

        # Use SUS to get the population
        temp_population = SelectionOperators.sus_selection_fast_clone(self.random_generator,
                                                                      self.chromosomes,
                                                                      number_of_children)

        # Elitism, add in the elitist individual
        temp_population[-1] = elite

        # For each chromosome, split in half and assign each to the child populations
        for ind in temp_population:
            temp = ind.split_chromosome()
            child_1.chromosomes.append(temp[0])
            child_2.chromosomes.append(temp[1])

        return child_1, child_2

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

    def check_if_improved(self):
        """
        Checks if there has been an improvement in the current score after evaluation compared to before evaluation.

        If an improvement is detected, sets the 'has_improved' flag to True.

        Returns:
            None
        """
        # Check if improvement hasn't been detected yet and if the current score has improved
        if not self.has_improved and self.elite.get_fitness() > self.best_fitness_at_creation:
            self.has_improved = True  # Set the flag to True to indicate improvement

        if not self.has_improved:  # If there isn't improvement, increase the merge tracker
            self.merge_tracker += 1

    def deep_clone(self, name, generation, parent_population):
        """
        Create a deep clone of the population and its chromosomes.

        Returns:
            Population: A deep clone of the population.
        """
        # Create a new Population instance with the same name, generation, and parent_population
        clone_population = Population(self.random_generator, name, generation, self.best_fitness_at_creation, parent_population)

        # Clone each chromosome within the population and add it to the clone_population
        for chromosome in self.chromosomes:
            clone_chromosome = chromosome.clone()  # Assuming Chromosome class has a clone method
            clone_population.add_chromosome(clone_chromosome)

        return clone_population

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

    def print_population_simple(self):
        """
        Print the details of each chromosome in the population.
        """
        if self.parent_population is not None:
            parent_name = self.parent_population.get_name()
        else:
            parent_name = self.get_name()
        # parent_name = self.parent_population.get_name() if self.parent_population is not None else self.get_name()
        #
        # parent_name = self.parent_population.get_name() if self.parent_population is not None else self.get_name()
        print(
            f"Population: {self.name}, created generation: {self.generation}, parent population: {parent_name}")
        for i, chromosome in enumerate(self.chromosomes):
            print(f"Chromosome {i + 1}:")
            chromosome.print_values_simple()
            print()

    def get_chromosome_with_min_fitness(self):
        """
        Get the chromosome with the lowest fitness score in the population.

        Returns:
            Chromosome: The chromosome with the lowest fitness score.
        """
        if not self.chromosomes:
            raise ValueError("The population is empty.")

        lowest_fitness_chromosome = self.chromosomes[0]
        lowest_fitness = lowest_fitness_chromosome.get_fitness()

        for chromosome in self.chromosomes:
            fitness = chromosome.get_fitness()
            if fitness < lowest_fitness:
                lowest_fitness = fitness
                lowest_fitness_chromosome = chromosome

        return lowest_fitness_chromosome

    def get_chromosome_with_max_fitness(self):
        """
        Get the chromosome with the highest fitness score in the population.

        Returns:
            Chromosome: The chromosome with the highest fitness score.
        """
        if not self.chromosomes:
            raise ValueError("The population is empty.")

        highest_fitness_chromosome = self.chromosomes[0]
        highest_fitness = highest_fitness_chromosome.get_fitness()

        for chromosome in self.chromosomes:
            fitness = chromosome.get_fitness()
            if fitness > highest_fitness:
                highest_fitness = fitness
                highest_fitness_chromosome = chromosome

        return highest_fitness_chromosome.clone()

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
