# Import custom fitness function
from evolutionary_algorithm.evaluation.fitness_function.rosenbrock import rosenbrock

chrom =  [-1.013774056826857,
1.0373538855323616,
1.0795403929627005,
1.1679396489767306 ]

print(rosenbrock(chrom))