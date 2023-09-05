import os
import csv
import sys


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
        sys.stdout.flush()

        # Set up file paths
        self.text_file = os.path.join(self.main_directory, 'results.txt')
        self.csv_fits = os.path.join(self.main_directory, 'fitness.csv')

        # Redirect standard output to the text file and flush to ensure immediate file creation
        sys.stdout = open(self.text_file, 'w')
        sys.stdout.flush()

        # Open the CSV file for fitness and population output and flush to ensure immediate file creation
        self.fits_file = open(self.csv_fits, 'w', newline='', encoding='utf-8')
        self.fit_writer = csv.writer(self.fits_file)
        self.fits_file.flush()

    def close(self) -> None:
        """
        Closes the output files.
        """
        # Close the files when done
        sys.stdout.close()
        self.fits_file.close()

    def write_fitness(self, best_fitness, avg_fitness, worst_fitness) -> None:
        """
        Writes fitness values to the fitness CSV file.

        Args:
            best_fitness (float): Best fitness value.
            avg_fitness (float): Average fitness value.
            worst_fitness (float): Worst fitness value.
        """
        self.fit_writer.writerow((best_fitness, avg_fitness, worst_fitness))

    def print_stats_short(self, population, fitness_scores) -> None:
        """
        Print summarized statistics about the fitness scores of a population.

        Args:
            population (list): List of individuals in the population.
            fitness_scores (list): List of fitness scores corresponding to the individuals.

        Returns:
            None
        """
        length = len(population)
        mean = sum(fitness_scores) / length
        sum2 = sum(x * x for x in fitness_scores)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        print("  Min %s" % min(fitness_scores))
        print("  Avg %s" % mean)
        print("  Max %s" % max(fitness_scores))
        print("  Std %s" % std)

        self.write_fitness(min(fitness_scores), mean, max(fitness_scores))

    def flush(self) -> None:
        """
        Flushes the output streams to their respective files.
        """
        sys.stdout.flush()
        self.fits_file.flush()
