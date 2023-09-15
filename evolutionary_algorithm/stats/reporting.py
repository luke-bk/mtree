import os
import csv
import sys
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

    def print_stats_short(self, population, fitness_scores, active_populations) -> None:
        """
        Print summarized statistics about the fitness scores of a population.

        Args:
            population (list): List of individuals in the population.
            fitness_scores (list): List of fitness scores corresponding to the individuals.
            active_populations (int): The number of active populations.

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
                print("In while loops")
                pops_reader = csv.reader(pops_file)
                for row in pops_reader:
                    print("In forloop")
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

            # Check the length (shape) of the generations list
            print("Length of generations list:", len(generations))

            # Check the length (shape) of the population_numbers list
            print("Length of max list:", len(min_fitness))

            # Check the length (shape) of the population_numbers list
            print("Length of population_numbers list:", len(population_numbers))

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

            # Save the plot in the results folder
            plot_file = os.path.join(self.main_directory, 'fitness_plot_with_target_and_population.png')
            plt.savefig(plot_file)
            plt.close()

        except Exception as e:
            print(f"Error plotting fitness: {e}")

    def flush(self) -> None:
        """
        Flushes the output streams to their respective files.
        """
        sys.stdout.flush()
        self.fits_file.flush()
        self.pops_file.flush()
