# import cv2
# import numpy as np
#
# from evolutionary_algorithm.chromosome.ChromosomeReal import ChromosomeReal
# from helpers.random_generator import RandomGenerator
# from evolutionary_algorithm.evaluation.fitness_function.manhattan_distance import manhattan_distance_fitness
#
# # "images/test_images/base_7.png"
# random_gen = RandomGenerator(seed=5)
#
# chromosome = ChromosomeReal(random_gen, "0", "../images/test_images/base_7.png")
#
# # print(type(chromosome.chromosome))
#
# # chromosome.display_image()
#
# a,b,c,d = chromosome.split_chromosome()
#
# aa, ab, ac, ad = a.split_chromosome()
#
# print(chromosome.chromosome.shape)
# print(a.chromosome.shape)
# print(b.chromosome.shape)
# print(c.chromosome.shape)
# print(d.chromosome.shape)
# print(aa.chromosome.shape)
# print(ab.chromosome.shape)
# print(ac.chromosome.shape)
# print(ad.chromosome.shape)
#
# target_image = cv2.imread("../images/test_images/base_7.png")
#
# # print (manhattan_distance_fitness(chromosome.chromosome, target_image))
# # print (manhattan_distance_fitness(target_image, target_image))
#
# aa.display_image()
# # b.display_image()
# # c.display_image()
# # d.display_image()
#
#
# top_half = np.hstack((a.chromosome, d.chromosome))
# bottom_half = np.hstack((b.chromosome, c.chromosome))
#
# # Vertically combine to get the full image
# full_image = np.vstack((top_half, bottom_half))
#
# a.chromosome = full_image
# a.display_image()
#
# # print (combined_image)
# # print(chromosome.chromosome)
# print (manhattan_distance_fitness(full_image, target_image))
#


import random

# # Define the dictionary keys
# keys = ["030", "0310", "0311", "0312", "0313", "032", "033", "04", "000", "001", "002", "003", "020", "021", "022", "023"]
#
# # Initialize an empty dictionary
# dict_with_arrays = {}
#
# # Iterate over the keys and fill the dictionary accordingly
# for key in keys:
#     # Determine the length of the array based on the length of the key
#     array_length = 8 if len(key) == 2 else 2
#
#     # Create an array of random 0s and 1s of the determined length
#     array = [random.randint(0, 1) for _ in range(array_length)]
#
#     # Assign the array to the dictionary key
#     dict_with_arrays[key] = array
#
# # Sort the dictionary by keys (from smallest to largest)
# sorted_dict_with_arrays = {key: dict_with_arrays[key] for key in sorted(dict_with_arrays)}
#
# # Printing the sorted dictionary
# for key, value in sorted_dict_with_arrays.items():
#     print(f"'{key}': {value}")
#
# while len(sorted_dict_with_arrays) > 1:
#     new_dict = {}
#
#     # Group keys by their parent key and merge their arrays
#     for key, array in sorted_dict_with_arrays.items():
#         parent_key = key[:-1]  # Parent key is all but the last character
#         if parent_key not in new_dict:
#             new_dict[parent_key] = []
#         new_dict[parent_key].extend(array)
#
#     # Update the dictionary to be sorted
#     sorted_dict_with_arrays = {key: new_dict[key] for key in sorted(new_dict)}
#     print ("-----------------------")
#     # Printing the sorted dictionary
#     for key, value in sorted_dict_with_arrays.items():
#         print(f"'{key}': {value}")
#
# # # Final output
# # final_key, final_array = list(sorted_dict_with_arrays.items())[0]
# # print(f"'{final_key}': {final_array}")

import random

# Sample dictionary as provided
dict_with_arrays = {
    '000': [1, 0],
    '001': [0, 1],
    '002': [0, 1],
    '003': [1, 0],
    '020': [1, 1],
    '021': [0, 0],
    '022': [0, 0],
    '023': [1, 0],
    '030': [0, 1],
    '0310': [1, 1],
    '0311': [0, 1],
    '0312': [1, 0],
    '0313': [1, 0],
    '032': [1, 0],
    '033': [0, 0],
    '04': [1, 0, 1, 0, 1, 1, 1, 0]
}

while len(dict_with_arrays) > 1:
    keys_to_remove = []
    sorted_keys = sorted(dict_with_arrays.keys(), key=lambda x: (-len(x), x))

    for key in sorted_keys:
        parent_key = key[:-1] if len(key) > 1 else '0'

        if parent_key not in dict_with_arrays:
            dict_with_arrays[parent_key] = dict_with_arrays[key].copy()
        else:
            dict_with_arrays[parent_key].extend(dict_with_arrays[key])

        if parent_key != key:
            keys_to_remove.append(key)

        print("-------------------")
        for k, value in dict_with_arrays.items():
            print(f"'{k}': {value}")

    # Remove the keys that were merged
    for key in keys_to_remove:
        del dict_with_arrays[key]

# Final output
print(dict_with_arrays)
