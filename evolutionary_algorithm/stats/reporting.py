import os
import csv
import sys

import os
import pydicom
import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, RLELossless

import cv2
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


class ExperimentResults:
    def __init__(self, seed, main_directory='../../results/'):
        """
        Initializes an instance of ExperimentResults.

        Args:
            seed (int): Seed value for the experiment.
            main_directory (str, optional): Main directory where experiment results will be stored.
                Defaults to '../../results/'.
        """
        self.seed = seed
        self.main_directory = main_directory

        # Create the new directory if it doesn't exist and flush to ensure immediate directory creation
        os.makedirs(self.main_directory, exist_ok=True)

        # Set up file paths
        self.text_file = os.path.join(self.main_directory, 'results.txt')
        self.csv_fits = os.path.join(self.main_directory, 'fitness.csv')
        self.csv_population_information = os.path.join(self.main_directory, 'populations.csv')

        # Redirect standard output to the text file
        self.stdout_orig = sys.stdout
        sys.stdout = open(self.text_file, 'w')

        # Open the CSV file for fitness and population output and flush to ensure immediate file creation
        self.fits_file = open(self.csv_fits, 'w', newline='', encoding='utf-8')
        self.fit_writer = csv.writer(self.fits_file)

        # Open the CSV file for population output and flush to ensure immediate file creation
        self.pops_file = open(self.csv_population_information, 'w', newline='', encoding='utf-8')
        self.pops_writer = csv.writer(self.pops_file)

    def close(self) -> None:
        """
        Closes the output files.
        """
        # Close the files when done
        self.flush()
        sys.stdout.close()
        self.fits_file.close()
        self.pops_file.close()

        # Restore the original standard output stream
        sys.stdout = self.stdout_orig

    def write_fitness(self, best_fitness, avg_fitness, worst_fitness) -> None:
        """
        Writes fitness values to the fitness CSV file.

        Args:
            best_fitness (float): Best fitness value.
            avg_fitness (float): Average fitness value.
            worst_fitness (float): Worst fitness value.
        """
        self.fit_writer.writerow((best_fitness, avg_fitness, worst_fitness))

    def write_number_populations(self, number_pops) -> None:
        """
        Writes number of populations to the pops CSV file.

        Args:
            number_pops (float): Number of active populations
        """
        self.pops_writer.writerow((number_pops,))

    def print_stats_short(self, evaluations, fitness_scores, active_populations, binary_tree) -> None:
        """
        Print summarized statistics about the fitness scores of a population.

        Args:
            evaluations : List of individuals in the population.
            fitness_scores (list): List of fitness scores corresponding to the individuals.
            active_populations (int): The number of active populations.
            binary_tree (BinaryTree): The structure of the populations.


        Returns:
            None
        """
        length = evaluations
        mean = sum(fitness_scores) / length
        sum2 = sum(x * x for x in fitness_scores)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        print("  Min %s" % min(fitness_scores))
        print("  Avg %s" % mean)
        print("  Max %s" % max(fitness_scores))
        print("  Std %s" % std)
        print("   ")

        for x in binary_tree.get_leaf([]):
            print(x.print_self())
        print("   ")

        self.write_fitness(min(fitness_scores), mean, max(fitness_scores))
        self.write_number_populations(number_pops=active_populations)

    def plot_fitness(self):
        """
        Plots fitness values from the fitness CSV file and saves the plot.

        Returns:
            None
        """
        try:
            # Read fitness values from the CSV file
            min_fitness = []
            avg_fitness = []
            max_fitnessw = []

            with open(self.csv_fits, 'r', newline='', encoding='utf-8') as fits_file:
                fit_reader = csv.reader(fits_file)
                for row in fit_reader:
                    min_fitness.append(float(row[0]))
                    avg_fitness.append(float(row[1]))
                    max_fitnessw.append(float(row[2]))

            # Create a line plot
            generations = list(range(1, len(min_fitness) + 1))
            plt.figure(figsize=(10, 6))
            plt.plot(generations, max_fitnessw, label='Max Fitness')
            plt.plot(generations, avg_fitness, label='Average Fitness')
            plt.plot(generations, min_fitness, label='Min Fitness')
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.title('Fitness Over Generations')
            plt.legend()

            # Save the plot in the results folder
            plot_file = os.path.join(self.main_directory, 'fitness_plot.png')
            plt.savefig(plot_file)
            plt.close()

            print(f"Fitness plot saved to {plot_file}")
        except Exception as e:
            print(f"Error plotting fitness: {e}")

    def plot_fitness_with_target(self, target_score=None):
        """
        Plots fitness values from the fitness CSV file and saves the plot and the optimal fitness score.
        Additionally, the generation where the score was achieved is also marked with a vertical line.

        Args:
            target_score (float, optional): Target fitness score to plot a horizontal line. Defaults to None.

        Returns:
            None
        """
        try:
            # Read fitness values from the CSV file
            min_fitness = []
            avg_fitness = []
            max_fitness = []

            with open(self.csv_fits, 'r', newline='', encoding='utf-8') as fits_file:
                fit_reader = csv.reader(fits_file)
                for row in fit_reader:
                    min_fitness.append(float(row[0]))
                    avg_fitness.append(float(row[1]))
                    max_fitness.append(float(row[2]))

            # Create a line plot
            generations = list(range(1, len(min_fitness) + 1))
            plt.figure(figsize=(10, 6))
            plt.plot(generations, max_fitness, label='Max Fitness')
            plt.plot(generations, avg_fitness, label='Average Fitness')
            plt.plot(generations, min_fitness, label='Min Fitness')

            if target_score is not None:
                # Plot a horizontal line at the target score
                plt.axhline(target_score, color='red', linestyle='--', label='Target Score', linewidth=0.25)

                # Check if the best fitness achieves the target score
                best_generation = None
                best_fitness_value = None
                for generation, fitness in enumerate(max_fitness, start=1):
                    if fitness >= target_score:
                        best_generation = generation
                        best_fitness_value = fitness
                        break

                if best_generation is not None:
                    # Plot a vertical line and annotate it with the best generation
                    plt.axvline(best_generation, color='green', linestyle='--', label='Achieved Target', linewidth=0.25)
                    plt.annotate(f'Gen {best_generation}\nScore {best_fitness_value:.2f}',
                                 xy=(best_generation, best_fitness_value),
                                 xytext=(best_generation + 10, best_fitness_value + 10),
                                 arrowprops=dict(arrowstyle='->', color='black', linewidth=0.25))

            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.title('Fitness Over Generations')
            plt.legend()

            # Save the plot in the results folder
            plot_file = os.path.join(self.main_directory, 'fitness_plot_with_target.png')
            plt.savefig(plot_file)
            plt.close()

            print(f"Fitness plot saved to {plot_file}")
        except Exception as e:
            print(f"Error plotting fitness: {e}")

    def plot_fitness_with_target_and_populations(self, target_score=None):
        """
        Plots fitness values from the fitness CSV file and saves the plot and the optimal fitness score.
        Additionally, the generation where the score was achieved is also marked with a vertical line, and the number
        of populations at each generation is tracked.

        Args:
            target_score (float, optional): Target fitness score to plot a horizontal line. Defaults to None.

        Returns:
            None
        """
        try:
            # Read fitness values from the CSV file
            min_fitness = []
            avg_fitness = []
            max_fitness = []

            with open(self.csv_fits, 'r', newline='', encoding='utf-8') as fits_file:
                fit_reader = csv.reader(fits_file)
                for row in fit_reader:
                    min_fitness.append(float(row[0]))
                    avg_fitness.append(float(row[1]))
                    max_fitness.append(float(row[2]))

            # Read population numbers from the CSV file
            population_numbers = []

            with open(self.csv_population_information, 'r', newline='', encoding='utf-8') as pops_file:
                pops_reader = csv.reader(pops_file)
                for row in pops_reader:
                    population_numbers.append(float(row[0]))

            # Create a line plot with twin y-axes
            generations = list(range(1, len(min_fitness) + 1))
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # Plot fitness values
            ax1.plot(generations, max_fitness, label='Max Fitness')
            ax1.plot(generations, avg_fitness, label='Average Fitness')
            ax1.plot(generations, min_fitness, label='Min Fitness')

            # Set labels for the left y-axis
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Fitness')
            ax1.set_title('Fitness Over Generations')
            ax1.legend()

            # Create a twin y-axis for population numbers
            ax2 = ax1.twinx()

            # Plot population numbers as transparent bars on the right y-axis
            ax2.bar(generations, population_numbers, alpha=0.1, color='blue', label='Population Numbers')

            ax2.set_ylabel('Number of Populations')
            ax2.yaxis.set_major_locator(MultipleLocator(1))

            # Add target score line and achievement marker if specified
            if target_score is not None:
                ax1.axhline(target_score, color='red', linestyle='--', label='Target Score', linewidth=0.25)

                best_generation = None
                best_fitness_value = None
                for generation, fitness in enumerate(max_fitness, start=1):
                    if fitness >= target_score:
                        best_generation = generation
                        best_fitness_value = fitness
                        break

                if best_generation is not None:
                    ax1.axvline(best_generation, color='green', linestyle='--', label='Achieved Target', linewidth=0.25)
                    ax1.annotate(f'Gen {best_generation}\nScore {best_fitness_value:.2f}',
                                 xy=(best_generation, best_fitness_value),
                                 xytext=(best_generation + 10, best_fitness_value + 10),
                                 arrowprops=dict(arrowstyle='->', color='black', linewidth=0.25))

            # Ensure the left y-axis starts at 0
            ax1.set_ylim(0, target_score + 5)  # Set the minimum value to 0 and the maximum value to max_value

            # Save the plot in the results folder
            plot_file = os.path.join(self.main_directory, 'fitness_plot_with_target_and_population.png')
            plt.savefig(plot_file)
            plt.close()

        except Exception as e:
            print(f"Error plotting fitness: {e}")

    def plot_fitness_with_target_and_populations_min_task(self, target_score=None):
        try:
            # Read fitness values from the CSV file
            min_fitness = []
            avg_fitness = []
            max_fitness = []

            with open(self.csv_fits, 'r', newline='', encoding='utf-8') as fits_file:
                fit_reader = csv.reader(fits_file)
                for row in fit_reader:
                    min_fitness.append(float(row[0]))
                    avg_fitness.append(float(row[1]))
                    max_fitness.append(float(row[2]))

            # Read population numbers from the CSV file
            population_numbers = []

            with open(self.csv_population_information, 'r', newline='', encoding='utf-8') as pops_file:
                pops_reader = csv.reader(pops_file)
                for row in pops_reader:
                    population_numbers.append(float(row[0]))

            # Create a line plot with twin y-axes
            generations = list(range(1, len(min_fitness) + 1))
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # Plot fitness values
            ax1.plot(generations, max_fitness, label='Max Fitness')
            ax1.plot(generations, avg_fitness, label='Average Fitness')
            ax1.plot(generations, min_fitness, label='Min Fitness')

            # Set labels for the left y-axis
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Fitness')
            ax1.set_title('Fitness Over Generations')
            ax1.legend()

            # Create a twin y-axis for population numbers
            ax2 = ax1.twinx()

            # Plot population numbers as transparent bars on the right y-axis
            ax2.bar(generations, population_numbers, alpha=0.1, color='blue', label='Population Numbers')

            ax2.set_ylabel('Number of Populations')
            ax2.yaxis.set_major_locator(MultipleLocator(1))

            # Add target score line and achievement marker if specified
            if target_score is not None:
                ax1.axhline(target_score, color='red', linestyle='--', label='Target Score', linewidth=0.25)

                best_generation = None
                best_fitness_value = None
                for generation, fitness in enumerate(max_fitness, start=1):
                    if fitness <= target_score:  # Change the condition to check for <=
                        best_generation = generation
                        best_fitness_value = fitness
                        break

                if best_generation is not None:
                    ax1.axvline(best_generation, color='green', linestyle='--', label='Achieved Target', linewidth=0.25)
                    ax1.annotate(f'Gen {best_generation}\nScore {best_fitness_value:.2f}',
                                 xy=(best_generation, best_fitness_value),
                                 xytext=(best_generation + 10, best_fitness_value + 10),
                                 arrowprops=dict(arrowstyle='->', color='black', linewidth=0.25))

            # Set the maximum y-axis limit based on the maximum fitness value
            max_fitness_value = max(max_fitness)
            ax1.set_ylim(0, max_fitness_value + 5)

            # Save the plot in the results folder
            plot_file = os.path.join(self.main_directory, 'fitness_plot_with_target_and_population.png')
            plt.savefig(plot_file)
            plt.close()

        except Exception as e:
            print(f"Error plotting fitness: {e}")

    def plot_fitness_with_target_and_populations_min_isolated(self, target_score=None):
        try:
            # Read fitness values from the CSV file
            min_fitness = []

            with open(self.csv_fits, 'r', newline='', encoding='utf-8') as fits_file:
                fit_reader = csv.reader(fits_file)
                for row in fit_reader:
                    min_fitness.append(float(row[0]))

            # Read population numbers from the CSV file
            population_numbers = []

            with open(self.csv_population_information, 'r', newline='', encoding='utf-8') as pops_file:
                pops_reader = csv.reader(pops_file)
                for row in pops_reader:
                    population_numbers.append(float(row[0]))

            # Create a line plot with twin y-axes
            generations = list(range(1, len(min_fitness) + 1))
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # Plot minimum fitness values
            ax1.plot(generations, min_fitness, label='Min Fitness', color='green')

            # Set labels for the left y-axis
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Fitness')
            ax1.set_title('Minimum Fitness Over Generations')
            ax1.legend()

            # Set appropriate y-axis limits based on fitness values
            ax1.set_ylim(min(min_fitness) - 100, max(min_fitness) + 100)  # Adjust the limits dynamically

            # Create a twin y-axis for population numbers
            ax2 = ax1.twinx()

            # Plot population numbers as transparent bars on the right y-axis
            ax2.bar(generations, population_numbers, alpha=0.1, color='blue', label='Population Numbers')

            ax2.set_ylabel('Number of Populations')
            ax2.yaxis.set_major_locator(MultipleLocator(1))

            # Add target score line and achievement marker if specified
            if target_score is not None:
                ax1.axhline(target_score, color='red', linestyle='--', label='Target Score', linewidth=0.25)

                best_generation = None
                best_fitness_value = None
                for generation, fitness in enumerate(min_fitness, start=1):
                    if fitness <= target_score:
                        best_generation = generation
                        best_fitness_value = fitness
                        break

                if best_generation is not None:
                    ax1.axvline(best_generation, color='green', linestyle='--', label='Achieved Target', linewidth=0.25)
                    ax1.annotate(f'Gen {best_generation}\nScore {best_fitness_value:.2f}',
                                 xy=(best_generation, best_fitness_value),
                                 xytext=(best_generation + 10, best_fitness_value + 10),
                                 arrowprops=dict(arrowstyle='->', color='black', linewidth=0.25))

            # Save the plot in the results folder
            plot_file = os.path.join(self.main_directory, 'fitness_plot_with_min_isolated.png')
            plt.savefig(plot_file)
            plt.close()

        except Exception as e:
            print(f"Error plotting fitness: {e}")


    def plot_fitness_with_target_and_populations_min_task_zoom(self, target_score=None):
        try:
            # Read fitness values from the CSV file
            min_fitness = []
            avg_fitness = []
            max_fitness = []

            with open(self.csv_fits, 'r', newline='', encoding='utf-8') as fits_file:
                fit_reader = csv.reader(fits_file)
                for row in fit_reader:
                    min_fitness.append(float(row[0]))
                    avg_fitness.append(float(row[1]))
                    max_fitness.append(float(row[2]))

            # Read population numbers from the CSV file
            population_numbers = []

            with open(self.csv_population_information, 'r', newline='', encoding='utf-8') as pops_file:
                pops_reader = csv.reader(pops_file)
                for row in pops_reader:
                    population_numbers.append(float(row[0]))

            # Create a line plot with twin y-axes
            generations = list(range(1, len(min_fitness) + 1))
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # Plot fitness values
            ax1.plot(generations, max_fitness, label='Max Fitness')
            ax1.plot(generations, avg_fitness, label='Average Fitness')
            ax1.plot(generations, min_fitness, label='Min Fitness')

            # Set labels for the left y-axis
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Fitness')
            ax1.set_title('Fitness Over Generations')
            ax1.legend()

            # Create a twin y-axis for population numbers
            ax2 = ax1.twinx()

            # Plot population numbers as transparent bars on the right y-axis
            ax2.bar(generations, population_numbers, alpha=0.1, color='blue', label='Population Numbers')

            ax2.set_ylabel('Number of Populations')
            ax2.yaxis.set_major_locator(MultipleLocator(1))

            # Add target score line and achievement marker if specified
            if target_score is not None:
                ax1.axhline(target_score, color='red', linestyle='--', label='Target Score', linewidth=0.25)

                best_generation = None
                best_fitness_value = None
                for generation, fitness in enumerate(max_fitness, start=1):
                    if fitness <= target_score:  # Change the condition to check for <=
                        best_generation = generation
                        best_fitness_value = fitness
                        break

                if best_generation is not None:
                    ax1.axvline(best_generation, color='green', linestyle='--', label='Achieved Target', linewidth=0.25)
                    ax1.annotate(f'Gen {best_generation}\nScore {best_fitness_value:.2f}',
                                 xy=(best_generation, best_fitness_value),
                                 xytext=(best_generation + 10, best_fitness_value + 10),
                                 arrowprops=dict(arrowstyle='->', color='black', linewidth=0.25))

            # Set the y-axis limit to be between 0 and 1
            ax1.set_ylim(0, 1)

            # Save the plot in the results folder
            plot_file = os.path.join(self.main_directory, 'fitness_plot_with_target_and_population_zoomed.png')
            plt.savefig(plot_file)
            plt.close()

        except Exception as e:
            print(f"Error plotting fitness: {e}")

    def find_best_solution(self, binary_tree):
        """
        Find the best solution among leaf nodes of a binary tree and print it.

        Args:
            binary_tree: The binary tree to search for the best solution.

        Returns:
            None
        """
        best_leaf_node = None
        best_fitness = None

        for leaf_node in binary_tree.get_leaf([]):
            elite = leaf_node.population.elite
            fitness = elite.get_fitness()

            if best_fitness is None or fitness > best_fitness:
                best_leaf_node = leaf_node
                best_fitness = fitness

        best_solution = best_leaf_node.population.elite_collaborators
        best_solution.insert(
            best_leaf_node.population.index_in_collaboration,
            best_leaf_node.population.elite
        )

        print(f"Best fitness of run: {best_fitness}")
        print(best_leaf_node.print_self())

        for chromosome in best_solution:
            print(chromosome.print_values_expressed())

    def find_best_solution_image(self, quad_tree):
        """
        Find the best solution among leaf nodes of a binary tree and print it.

        Args:
            binary_tree: The binary tree to search for the best solution.

        Returns:
            None
        """
        best_leaf_node = None
        best_fitness = None
        best_elite = quad_tree.get_leaf([])[0].population.elite

        for leaf_node in quad_tree.get_leaf([]):
            elite = leaf_node.population.elite
            fitness = elite.get_fitness()
            # elite = best_elite

            if best_fitness is None or fitness < best_fitness:
                best_leaf_node = leaf_node
                best_fitness = fitness
                best_elite = leaf_node.population.elite

        print(f"Best fitness of run: {best_fitness}")
        print(best_leaf_node.print_self())

        plot_file = os.path.join(self.main_directory, 'best_chromosome_evolved.png')
        cv2.imwrite(plot_file, best_elite.chromosome)



    def save_as_dicom(self, image_array, directory, base_image):
        """
        Saves a numpy array as a DICOM file.

        Args:
            image_array (numpy.ndarray): The image to save.
            directory (str): The directory to save the DICOM file.
        """
        # Create a new DICOM file

        ds = pydicom.dcmread(base_image)
        ds.PixelData = image_array.astype('uint16')
        ds.compress(RLELossless, image_array)
        dicom_file = os.path.join(directory, 'best_chromosome_evolved_duplicate_meta.dcm')
        ds.save_as(dicom_file)

    def save_as_dicom_copy_original(self, original_dcm_path, image_array, directory):
        """
        Saves a numpy array as a DICOM file, copying metadata from an original DICOM file.

        Args:
            original_dcm_path (str): Path to the original DICOM file.
            image_array (numpy.ndarray): The image to save.
            directory (str): The directory to save the DICOM file.
        """
        # Load the original DICOM file
        original_ds = pydicom.dcmread(original_dcm_path)

        # Update the pixel data with the new image
        original_ds.PixelData = image_array.tobytes()

        # Generate new UIDs for the modified image
        original_ds.SOPInstanceUID = pydicom.uid.generate_uid()
        original_ds.StudyInstanceUID = pydicom.uid.generate_uid()
        original_ds.SeriesInstanceUID = pydicom.uid.generate_uid()

        # Save the modified file
        dicom_file = os.path.join(directory, 'best_chromosome_evolved_duplicate_meta.dcm')
        pydicom.dcmwrite(dicom_file, original_ds)

        print(f"Saved DICOM file at {dicom_file}")


    def find_best_solution_image_dcm(self, quad_tree, base_image):
        """
        Find the best solution among leaf nodes of a quad tree and print it.

        Args:
            quad_tree: The quad tree to search for the best solution.

        Returns:
            None
        """
        best_leaf_node = None
        best_fitness = None
        best_elite = quad_tree.get_leaf([])[0].population.elite

        for leaf_node in quad_tree.get_leaf([]):
            elite = leaf_node.population.elite
            fitness = elite.get_fitness()
            # elite = best_elite

            if best_fitness is None or fitness < best_fitness:
                best_leaf_node = leaf_node
                best_fitness = fitness
                best_elite = leaf_node.population.elite

        print(f"Best fitness of run: {best_fitness}")
        print(best_leaf_node.print_self())

        # Save the best chromosome as a DICOM file
        self.save_as_dicom(best_elite.chromosome, self.main_directory, base_image)

        # Save the best chromosome as a DICOM file with duplicated meta
        # self.save_as_dicom_copy_original(original_path, quad_tree.population.elite.chromosome, self.main_directory)


    def flush(self) -> None:
        """
        Flushes the output streams to their respective files.
        """
        sys.stdout.flush()
        self.fits_file.flush()
        self.pops_file.flush()
