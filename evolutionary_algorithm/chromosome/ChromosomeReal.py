import copy
import matplotlib.pyplot as plt
from typing import List, Optional
from helpers.image_manipulation import add_clustered_noise_random_ellipse_fast_to_array, \
    add_clustered_noise_same_colour_profile_to_array, add_clustered_noise_to_grayscale_image, add_clustered_noise_to_grayscale_image_dcm


class ChromosomeReal:
    """
    A class representing a chromosome, which is an ndarray.

    Attributes:
        name (str): The name of the chromosome.
    """

    def __init__(
            self,
            random_generator,
            parent_name: str,
            noise_base_image_path: str,
            image_type: str
    ) -> None:
        """
        Constructor that initializes the Chromosome instance.

        Args:
            parent_name (str): The name of its parent..if its been split
            chromosome_length (int): The length of each part chromosome.
            noise_base_image_path (str): The folder location of the image
            gene_min (float, optional): The minimum value for real-valued genes. Defaults to None.
            gene_max (float, optional): The maximum value for real-valued genes. Defaults to None.
        """
        self.name = ""
        self.fitness = None
        self.set_name()  # set the unique name
        self.parent_name = parent_name  # Set the parent name provided when the constructor is called
        self.gene_type = "Real"  # What type of genes this chromosomes has
        # Create a chromosome as an ndarray with real numbers
        if image_type == "dcm":
            self.chromosome = add_clustered_noise_to_grayscale_image_dcm(image_to_change_path=noise_base_image_path,
                                                                     random_generator=random_generator,
                                                                     num_of_clusters=4)
        else:
            self.chromosome = add_clustered_noise_to_grayscale_image(image_to_change_path=noise_base_image_path,
                                                                           random_generator=random_generator,
                                                                           num_of_clusters=4)
        self.length = len(self.chromosome)

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
        Split the chromosome (image) into four quadrants.

        :return: A tuple of four 2D ndarrays, each representing a quadrant of the image.
        """
        if self.chromosome.ndim not in [2, 3]:
            print(self.chromosome.ndim)
            raise ValueError("Chromosome is not a 2D or 3D array.")

        top_left = self.clone()
        top_right = self.clone()
        bottom_left = self.clone()
        bottom_right = self.clone()

        rows, cols = self.chromosome.shape[:2]
        mid_row, mid_col = rows // 2, cols // 2

        # Define quadrants for both 2D and 3D arrays
        if self.chromosome.ndim == 2:  # Grayscale image
            top_left.chromosome = self.chromosome[:mid_row, :mid_col]
            top_right.chromosome = self.chromosome[:mid_row, mid_col:]
            bottom_left.chromosome = self.chromosome[mid_row:, :mid_col]
            bottom_right.chromosome = self.chromosome[mid_row:, mid_col:]
        else:  # Color image
            top_left.chromosome = self.chromosome[:mid_row, :mid_col, :]
            top_right.chromosome = self.chromosome[:mid_row, mid_col:, :]
            bottom_left.chromosome = self.chromosome[mid_row:, :mid_col, :]
            bottom_right.chromosome = self.chromosome[mid_row:, mid_col:, :]

        return top_left, bottom_left, bottom_right, top_right

    def _deep_copy_chromosome(self, chromosome):
        """
        Create a deep copy of the given chromosome.

        :param chromosome: The chromosome to create a deep copy of.
        :return: A deep copy of the chromosome.
        """
        copy_genes = chromosome[:]  # This creates a shallow copy of the list, which is sufficient for floats

        copy_instance = copy.deepcopy(self)  # Deep copy the actual Chromosome instance
        copy_instance.set_name()  # Rename the chromosome
        copy_instance.chromosome = copy_genes  # Assign the copied list of genes
        return copy_instance

    def merge_chromosome(self, input_chromosome) -> None:
        """
        Merges this chromosome, with the input into one.

        """
        self.chromosome = input_chromosome

    def clone(self) -> 'Chromosome':
        """
        Create a deep copy of the chromosome.

        Returns:
            Chromosome: A deep copy of the chromosome.
        """
        copy_instance = copy.deepcopy(self)  # Deep copy the actual part chromosome
        copy_instance.set_name()  # Rename the chromosome

        return copy_instance

    def print_values(self) -> None:
        """
        Print the values of the part chromosomes in the chromosome.
        """
        print(f"Chromosome {self.name}:")
        for i, part_chromosome in enumerate(self.chromosome):
            print(end="    ")
            print(f"Part Chromosome {i + 1} ({part_chromosome.name}):")
            part_chromosome.print_values()

    def print_values_simple(self) -> None:
        """
        Print the values of the part chromosomes in the chromosome.
        """
        for i, part_chromosome in enumerate(self.chromosome):
            print(end="    ")
            part_chromosome.print_values_simple()

    def print_values_verbose(self) -> None:
        """
        Print the values of the part chromosomes in the chromosome.
        """
        print(f"Chromosome {self.name}:")
        for i, part_chromosome in enumerate(self.chromosome):
            print(end="    ")
            print(f"Part Chromosome {i + 1} ({part_chromosome.name}):")
            part_chromosome.print_values_verbose()

    def display_image(self):
        """
        Display the chromosome as an image.
        """
        if self.chromosome.ndim == 2:  # Grayscale image
            plt.imshow(self.chromosome, cmap='gray')
        elif self.chromosome.ndim == 3:  # Color image
            plt.imshow(self.chromosome)
        else:
            raise ValueError("Chromosome does not have a valid image format.")
        plt.axis('off')  # No axis for images
        plt.show()
