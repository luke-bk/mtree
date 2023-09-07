import os
import csv
import sys
import matplotlib.pyplot as plt


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

        # Redirect standard output to the text file
        self.stdout_orig = sys.stdout
        sys.stdout = open(self.text_file, 'w')

        # Open the CSV file for fitness and population output and flush to ensure immediate file creation
        self.fits_file = open(self.csv_fits, 'w', newline='', encoding='utf-8')
        self.fit_writer = csv.writer(self.fits_file)

    def close(self) -> None:
        """
        Closes the output files.
        """
        # Close the files when done
        sys.stdout.close()
        self.fits_file.close()

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

    def flush(self) -> None:
        """
        Flushes the output streams to their respective files.
        """
        sys.stdout.flush()
        self.fits_file.flush()
