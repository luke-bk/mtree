from evolutionary_algorithm.Chromosome import Chromosome

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    chromosome = Chromosome(-5, -1, 1)

    chromosome.print_values()
    chromosome.print_values_verbose()

    for _ in range(10):
        print("in")

    print("out")