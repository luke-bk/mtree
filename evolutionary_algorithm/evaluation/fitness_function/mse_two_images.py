import numpy as np
import cv2


def mse(image_one, image_two) -> float:
    """
    Calculate the Mean Squared Error (MSE) between two images.
    The 'Mean Squared Error' between the two images is the sum of the squared difference between the two images.
    NOTE: the two images must have the same dimension

    Args:
        image_one (numpy.ndarray): First input image.
        image_two (numpy.ndarray): Second input image.

    Returns:
        float: Calculated MSE value.
    """
    # Compute the squared difference between the pixel values of the two images
    squared_diff = (image_one.astype("float") - image_two.astype("float")) ** 2
    # Sum up the squared differences
    err = np.sum(squared_diff)
    # Calculate the mean squared error by dividing by the number of pixels
    err /= float(image_one.shape[0] * image_one.shape[1])
    return err


def compare_images(image_one, image_two) -> float:
    """
    Compare two images using Mean Squared Error (MSE).

    Args:
        image_one (numpy.ndarray): First input image.
        image_two (numpy.ndarray): Second input image.

    Returns:
        float: MSE value indicating image similarity (lower values are more similar).
    """
    # Calculate the Mean Squared Error between the two images
    return mse(image_one, image_two)


def result(filepath_a, filepath_b) -> float:
    """
    Compare two images using Mean Squared Error (MSE).

    Args:
        filepath_a (str): Filepath to the first image.
        filepath_b (str): Filepath to the second image.

    Returns:
        float: MSE value indicating image similarity (lower values are more similar).
    """
    # Read the input images from file
    image_a = cv2.imread(filepath_a)
    image_b = cv2.imread(filepath_b)

    # Compare the images using Mean Squared Error
    return compare_images(image_a, image_b)
