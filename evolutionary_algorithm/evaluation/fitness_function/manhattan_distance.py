import numpy as np


def manhattan_distance_fitness(image_one, image_two):
    """
    Calculate the Manhattan distance between two images based on their RGB values.

    Args:
        image_one (numpy.ndarray): First input image.
        image_two (numpy.ndarray): Second input image.

    Returns:
        float: Calculated Manhattan distance.
    """
    # Ensure both images have the same dimensions
    if image_one.shape != image_two.shape:
        raise ValueError("Images must have the same dimensions.")

    # Calculate the Manhattan distance
    distance = np.sum(np.abs(image_one - image_two))
    return distance,
