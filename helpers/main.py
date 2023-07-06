from evolutionary_algorithm.Chromosome import Chromosome

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    chromosome = Chromosome("lol", 10, -1, 1)

    chromosome.print_values()
    chromosome.print_values_verbose()
