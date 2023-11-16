import cv2
import numpy as np

from evolutionary_algorithm.chromosome.ChromosomeReal import ChromosomeReal
from helpers.random_generator import RandomGenerator
from evolutionary_algorithm.evaluation.fitness_function.manhattan_distance import manhattan_distance_fitness

# "images/test_images/base_7.png"
random_gen = RandomGenerator(seed=5)

chromosome = ChromosomeReal(random_gen, "0", "../images/test_images/base_7.png")

# print(type(chromosome.chromosome))

# chromosome.display_image()

a,b,c,d = chromosome.split_chromosome()

aa, ab, ac, ad = a.split_chromosome()

print(chromosome.chromosome.shape)
print(a.chromosome.shape)
print(b.chromosome.shape)
print(c.chromosome.shape)
print(d.chromosome.shape)
print(aa.chromosome.shape)
print(ab.chromosome.shape)
print(ac.chromosome.shape)
print(ad.chromosome.shape)

target_image = cv2.imread("../images/test_images/base_7.png")

# print (manhattan_distance_fitness(chromosome.chromosome, target_image))
# print (manhattan_distance_fitness(target_image, target_image))

aa.display_image()
# b.display_image()
# c.display_image()
# d.display_image()


top_half = np.hstack((a.chromosome, d.chromosome))
bottom_half = np.hstack((b.chromosome, c.chromosome))

# Vertically combine to get the full image
full_image = np.vstack((top_half, bottom_half))

a.chromosome = full_image
a.display_image()

# print (combined_image)
# print(chromosome.chromosome)
print (manhattan_distance_fitness(full_image, target_image))

