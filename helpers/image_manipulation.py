import cv2
import numpy
import numpy as np
from matplotlib import pyplot as plt


def image_save_interval(max_generations, images_to_save):
    """
    Calculate the interval at which to save images during evolution.

    Args:
        max_generations (int): The maximum number of generations.
        images_to_save (int): The number of images to save.


    Returns:
        int: The interval at which to save images.
    """
    return max_generations // images_to_save


def get_unique_colors(image_path) -> numpy.array:
    """
    Retrieves the unique colors present in an image.

    Args:
        image_path (str): Path to the input image.

    Returns:
        numpy.ndarray: Array of unique colors present in the image.
    """
    # Read the image
    image = cv2.imread(image_path)

    # Reshape the image to a 2D array of pixels and their color channels,
    # then find the unique rows (colors)
    unique_colors = np.unique(image.reshape(-1, image.shape[2]), axis=0)

    return unique_colors  # Return the array of unique colors

# def add_clustered_noise_random_ellipse_fast_to_array_same_pallet(image_to_change_path, random_generator, min_intensity,
#                                                      max_intensity, min_ellipse_length, max_ellipse_length,
#                                                      min_ellipse_width, max_ellipse_width, min_changes, max_changes):
#     """
#     Adds clustered elliptical noise to an image, based on the original pixel value.
#
#     Args:
#         max_changes:
#         min_changes:
#         image_to_change_path (str): Path to the input image.
#             The path to the image that will NOT have clustered noise added to it (base image).
#
#         random_generator (np.random.Generator): Random number generator.
#             An instance of NumPy's random number generator used for consistent randomness across all calls in this
#             project for experiment running purposes.
#
#         min_intensity (int): Minimum intensity change.
#             The lower bound for the intensity change applied to each pixel within the noise clusters.
#
#         max_intensity (int): Maximum intensity change.
#             The upper bound for the intensity change applied to each pixel within the noise clusters.
#
#         min_ellipse_length (int): Minimum length of the major axis of the elliptical cluster.
#             The lower bound for the length of the major axis of the elliptical noise clusters.
#
#         max_ellipse_length (int): Maximum length of the major axis of the elliptical cluster.
#             The upper bound for the length of the major axis of the elliptical noise clusters.
#
#         min_ellipse_width (int): Minimum width of the elliptical cluster.
#             The lower bound for the width of the elliptical noise clusters, controlling their narrowness.
#
#         max_ellipse_width (int): Maximum width of the elliptical cluster.
#             The upper bound for the width of the elliptical noise clusters, controlling their width variation.
#
#         min_changes (int): Smallest random number of changes to make.
#
#         max_changes(int): Largest random number of changes to make
#
#     Returns:
#         numpy.ndarray: Noisy image with clustered noise while preserving color profile.
#             An array representing the noisy image with clustered noise added, maintaining the original color profile.
#     """
#     # Load the input image
#     image = cv2.imread(image_to_change_path)
#
#     # Create a copy of the original image to work with
#     noisy_image = image.copy()
#
#     # Generate random parameters for the elliptical noise cluster
#     center_x = random_generator.randint(0, image.shape[1])  # Random x-coordinate for cluster center
#     center_y = random_generator.randint(0, image.shape[0])  # Random y-coordinate for cluster center
#     major_axis = random_generator.randint(min_ellipse_length, max_ellipse_length)  # Random major axis length
#     minor_axis = random_generator.randint(min_ellipse_width, max_ellipse_width)  # Random minor axis length
#     angle = random_generator.randint(0, 180)  # Random rotation angle for the cluster
#
#     # Create an initial mask filled with zeros
#     mask = np.zeros(image.shape[:2], dtype=np.uint8)
#
#     # Draw an ellipse on the mask based on the generated parameters
#     cv2.ellipse(mask, (center_x, center_y), (major_axis, minor_axis), angle, 0, 360, 255, -1)
#
#     # Find the coordinates of non-zero pixels in the mask
#     non_zero_pixels = np.column_stack(np.where(mask))
#
#     # Generate a random number of changes to apply within the specified range
#     max_changes = random_generator.randint(min_changes, max_changes)
#     applied_changes = 0  # Counter for tracking the number of changes applied
#
#     # Get unique colors from the original image
#     unique_colors = get_unique_colors(image_to_change_path)
#
#     # Loop over non-zero pixels and apply changes until the maximum number of changes is reached
#     for row, col in non_zero_pixels:
#         if applied_changes >= max_changes:
#             break  # Exit the loop once the maximum number of changes has been reached
#
#         # Clip pixel coordinates to image boundaries
#         new_row = np.clip(row, 0, noisy_image.shape[0] - 1)
#         new_col = np.clip(col, 0, noisy_image.shape[1] - 1)
#
#         # Generate a random intensity change for each color channel
#         intensity_change = random_generator.randint(min_intensity, max_intensity, size=3)
#
#         # Add the intensity change to the pixel color while clipping values to the [0, 255] range
#         # noisy_color = np.clip(noisy_image[new_row, new_col] + intensity_change, 0, 255)
#         noisy_color = unique_colors[random_generator.choice(len(unique_colors),
#                                                                  size=cluster.shape[0] * cluster.shape[1])]
#
#         # Apply the new color to the noisy image
#         noisy_image[new_row, new_col] = noisy_color
#
#         # Increment the counter for applied changes
#         applied_changes += 1
#
#     # Return the modified noisy image
#     return noisy_image

def get_unique_grayscale_values(image):
    """
    Retrieves the unique grayscale values present in an image.

    Args:
        image (numpy.ndarray): Grayscale image.

    Returns:
        numpy.ndarray: Array of unique grayscale values present in the image.
    """
    # Find the unique values (intensities) in the image
    unique_values = np.unique(image)

    return unique_values

def add_clustered_noise_to_grayscale_image(image_to_change_path, random_generator, num_of_clusters):
    """
    Adds clustered noise to a grayscale image.

    Args:
        image_to_change_path (str): Path to the input grayscale image.
        random_generator: Random generator for noise application.
        num_of_clusters: Number of noise clusters.

    Returns:
        numpy.ndarray: Noisy grayscale image.
    """
    # Read the original image in grayscale
    image = cv2.imread(image_to_change_path, cv2.IMREAD_GRAYSCALE)

    # Initialize the noisy image with a copy of the original image
    noisy_image = image.copy()

    # Define the number of clusters and their positions
    cluster_positions = random_generator.randint(2, high=image.shape[0] - 2, size=(num_of_clusters, 2))

    # Get unique grayscale values from the original image
    unique_values = get_unique_grayscale_values(image)

    # Apply noise using sampled values from the unique_values array
    for row, col in cluster_positions:
        # Define the cluster area
        cluster = image[row - 2: row + 2, col - 2: col + 2]
        if cluster.size > 0:  # Check if cluster is not empty
            noise_values = unique_values[random_generator.choice(len(unique_values),
                                                                 size=cluster.size)]
            noisy_cluster = noise_values.reshape(cluster.shape).astype(np.uint8)
            noisy_image[row - 2: row + 2, col - 2: col + 2] = noisy_cluster

    return noisy_image  # Return the noisy grayscale image as a numpy array


def add_clustered_noise_same_colour_profile_to_array(image_to_change_path, random_generator, num_of_clusters):
    """
    Adds clustered noise to an image while maintaining its color profile.

    Args:
        num_of_clusters: How many clusters of noise do we want?
        random_generator: Same random generator for everything
        image_to_change_path (str): Path to the input image.

    Returns:
        numpy.ndarray: Noisy image with clustered noise while preserving color profile.
    """
    # Read the original image
    image = cv2.imread(image_to_change_path)

    # Initialize the noisy image with a copy of the original image
    noisy_image = image.copy()

    # Define the number of clusters and their positions
    cluster_positions = random_generator.randint(2, high=image.shape[0] - 2, size=(num_of_clusters, 2))

    # Get unique colors from the original image
    unique_colors = get_unique_colors(image_to_change_path)

    # Apply noise using sampled colors from the unique_colors array
    for row, col in cluster_positions:
        cluster = image[row - 2: row + 2, col - 2: col + 2]
        if cluster.size > 0:  # Check if cluster is not empty
            noise_colors = unique_colors[random_generator.choice(len(unique_colors),
                                                                 size=cluster.shape[0] * cluster.shape[1])]
            noisy_cluster = noise_colors.reshape(cluster.shape).astype(np.uint8)
            noisy_image[row - 2: row + 2, col - 2: col + 2] = noisy_cluster

    return noisy_image  # Return the noisy image as a numpy array

def add_clustered_noise_random_ellipse_fast_to_array(image_to_change_path, random_generator, min_intensity,
                                                     max_intensity, min_ellipse_length, max_ellipse_length,
                                                     min_ellipse_width, max_ellipse_width, min_changes, max_changes):
    """
    Adds clustered elliptical noise to an image, based on the original pixel value.

    Args:
        max_changes:
        min_changes:
        image_to_change_path (str): Path to the input image.
            The path to the image that will NOT have clustered noise added to it (base image).

        random_generator (np.random.Generator): Random number generator.
            An instance of NumPy's random number generator used for consistent randomness across all calls in this
            project for experiment running purposes.

        min_intensity (int): Minimum intensity change.
            The lower bound for the intensity change applied to each pixel within the noise clusters.

        max_intensity (int): Maximum intensity change.
            The upper bound for the intensity change applied to each pixel within the noise clusters.

        min_ellipse_length (int): Minimum length of the major axis of the elliptical cluster.
            The lower bound for the length of the major axis of the elliptical noise clusters.

        max_ellipse_length (int): Maximum length of the major axis of the elliptical cluster.
            The upper bound for the length of the major axis of the elliptical noise clusters.

        min_ellipse_width (int): Minimum width of the elliptical cluster.
            The lower bound for the width of the elliptical noise clusters, controlling their narrowness.

        max_ellipse_width (int): Maximum width of the elliptical cluster.
            The upper bound for the width of the elliptical noise clusters, controlling their width variation.

        min_changes (int): Smallest random number of changes to make.

        max_changes(int): Largest random number of changes to make

    Returns:
        numpy.ndarray: Noisy image with clustered noise while preserving color profile.
            An array representing the noisy image with clustered noise added, maintaining the original color profile.
    """
    # Load the input image
    image = cv2.imread(image_to_change_path)

    # Create a copy of the original image to work with
    noisy_image = image.copy()

    # Generate random parameters for the elliptical noise cluster
    center_x = random_generator.randint(0, image.shape[1])  # Random x-coordinate for cluster center
    center_y = random_generator.randint(0, image.shape[0])  # Random y-coordinate for cluster center
    major_axis = random_generator.randint(min_ellipse_length, max_ellipse_length)  # Random major axis length
    minor_axis = random_generator.randint(min_ellipse_width, max_ellipse_width)  # Random minor axis length
    angle = random_generator.randint(0, 180)  # Random rotation angle for the cluster

    # Create an initial mask filled with zeros
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Draw an ellipse on the mask based on the generated parameters
    cv2.ellipse(mask, (center_x, center_y), (major_axis, minor_axis), angle, 0, 360, 255, -1)

    # Find the coordinates of non-zero pixels in the mask
    non_zero_pixels = np.column_stack(np.where(mask))

    # Generate a random number of changes to apply within the specified range
    max_changes = random_generator.randint(min_changes, max_changes)
    applied_changes = 0  # Counter for tracking the number of changes applied

    # Loop over non-zero pixels and apply changes until the maximum number of changes is reached
    for row, col in non_zero_pixels:
        if applied_changes >= max_changes:
            break  # Exit the loop once the maximum number of changes has been reached

        # Clip pixel coordinates to image boundaries
        new_row = np.clip(row, 0, noisy_image.shape[0] - 1)
        new_col = np.clip(col, 0, noisy_image.shape[1] - 1)

        # Generate a random intensity change for each color channel
        intensity_change = random_generator.randint(min_intensity, max_intensity, size=3)

        # Add the intensity change to the pixel color while clipping values to the [0, 255] range
        noisy_color = np.clip(noisy_image[new_row, new_col] + intensity_change, 0, 255)

        # Apply the new color to the noisy image
        noisy_image[new_row, new_col] = noisy_color

        # Increment the counter for applied changes
        applied_changes += 1

    # Return the modified noisy image
    return noisy_image

def add_clustered_noise_random_ellipse_to_array(image_to_change_path, random_generator, num_clusters,
                                                min_cluster_density, max_cluster_density, min_intensity, max_intensity,
                                                min_pix_neighborhood, max_pix_neighborhood, min_ellipse_length,
                                                max_ellipse_length, min_ellipse_width, max_ellipse_width):
    """
    Adds clustered elliptical noise to an image, based on the original pixel value.

    Args:
        image_to_change_path (str): Path to the input image.
            The path to the image that will NOT have clustered noise added to it (base image).

        random_generator (np.random.Generator): Random number generator.
            An instance of NumPy's random number generator used for consistent randomness across all calls in this
            project for experiment running purposes.

        num_clusters (int): Number of clusters of noise to add.
            The total number of noise clusters to be added to the image.

        min_cluster_density (int): Minimum cluster density.
            The lower bound for the number of pixels in a cluster that will be modified with noise.

        max_cluster_density (int): Maximum cluster density.
            The upper bound for the number of pixels in a cluster that will be modified with noise.

        min_intensity (int): Minimum intensity change.
            The lower bound for the intensity change applied to each pixel within the noise clusters.

        max_intensity (int): Maximum intensity change.
            The upper bound for the intensity change applied to each pixel within the noise clusters.

        min_pix_neighborhood (int): Minimum pixel neighborhood offset.
            The lower bound for the offset used to select neighboring pixels for applying noise.

        max_pix_neighborhood (int): Maximum pixel neighborhood offset.
            The upper bound for the offset used to select neighboring pixels for applying noise.

        min_ellipse_length (int): Minimum length of the major axis of the elliptical cluster.
            The lower bound for the length of the major axis of the elliptical noise clusters.

        max_ellipse_length (int): Maximum length of the major axis of the elliptical cluster.
            The upper bound for the length of the major axis of the elliptical noise clusters.

        min_ellipse_width (int): Minimum width of the elliptical cluster.
            The lower bound for the width of the elliptical noise clusters, controlling their narrowness.

        max_ellipse_width (int): Maximum width of the elliptical cluster.
            The upper bound for the width of the elliptical noise clusters, controlling their width variation.

    Returns:
        numpy.ndarray: Noisy image with clustered noise while preserving color profile.
            An array representing the noisy image with clustered noise added, maintaining the original color profile.
    """
    image = cv2.imread(image_to_change_path)

    # Initialize the noisy image with a copy of the original image
    noisy_image = image.copy()

    # Apply noise by slightly changing colors of random non-black points within the cluster
    for _ in range(num_clusters):
        center_x = random_generator.randint(0, image.shape[1])
        center_y = random_generator.randint(0, image.shape[0])
        # Adjusts length of the major axis of the elliptical cluster.
        major_axis = random_generator.randint(min_ellipse_length, max_ellipse_length)
        # Controls how narrow the  cluster is along its shortest dimension
        minor_axis = random_generator.randint(min_ellipse_width, max_ellipse_width)
        angle = random_generator.randint(0, 180)  # Define the rotation of the elliptical cluster

        mask = np.zeros(image.shape[:2], dtype=np.uint8)  # initial mask filled with zeros, subsequent code will modify
        cv2.ellipse(mask, (center_x, center_y), (major_axis, minor_axis), angle, 0, 360, 255, -1)

        non_zero_pixels = np.column_stack(np.where(mask))
        for row, col in non_zero_pixels:
            pixel = noisy_image[row, col]
            if not np.all(pixel == 0):  # Skip black pixels
                # Increase this range for denser clusters
                num_pixels_to_change = random_generator.randint(min_cluster_density,
                                                                max_cluster_density)
                for _ in range(num_pixels_to_change):
                    # Offsets are used to select neighboring pixels for applying noise.
                    # Adjust this range for pixel neighborhood
                    rand_row = random_generator.randint(min_pix_neighborhood,
                                                        max_pix_neighborhood)
                    # Adjust this range for pixel neighborhood
                    rand_col = random_generator.randint(min_pix_neighborhood,
                                                        max_pix_neighborhood)
                    new_row = np.clip(row + rand_row, 0, noisy_image.shape[0] - 1)
                    new_col = np.clip(col + rand_col, 0, noisy_image.shape[1] - 1)
                    noisy_color = np.clip(noisy_image[new_row, new_col] + random_generator.randint(min_intensity,
                                                                                                   max_intensity,
                                                                                                   size=3), 0, 255)
                    noisy_image[new_row, new_col] = noisy_color

    return noisy_image  # Return the noisy image as a numpy array


def add_clustered_noise_random_ellipse_fast_to_array_same_pallet(image_to_change_path, random_generator, min_intensity,
                                                     max_intensity, min_ellipse_length, max_ellipse_length,
                                                     min_ellipse_width, max_ellipse_width, min_changes, max_changes):
    """
    Adds clustered elliptical noise to an image, based on the original pixel value.

    Args:
        max_changes:
        min_changes:
        image_to_change_path (str): Path to the input image.
            The path to the image that will NOT have clustered noise added to it (base image).

        random_generator (np.random.Generator): Random number generator.
            An instance of NumPy's random number generator used for consistent randomness across all calls in this
            project for experiment running purposes.

        min_intensity (int): Minimum intensity change.
            The lower bound for the intensity change applied to each pixel within the noise clusters.

        max_intensity (int): Maximum intensity change.
            The upper bound for the intensity change applied to each pixel within the noise clusters.

        min_ellipse_length (int): Minimum length of the major axis of the elliptical cluster.
            The lower bound for the length of the major axis of the elliptical noise clusters.

        max_ellipse_length (int): Maximum length of the major axis of the elliptical cluster.
            The upper bound for the length of the major axis of the elliptical noise clusters.

        min_ellipse_width (int): Minimum width of the elliptical cluster.
            The lower bound for the width of the elliptical noise clusters, controlling their narrowness.

        max_ellipse_width (int): Maximum width of the elliptical cluster.
            The upper bound for the width of the elliptical noise clusters, controlling their width variation.

        min_changes (int): Smallest random number of changes to make.

        max_changes(int): Largest random number of changes to make

    Returns:
        numpy.ndarray: Noisy image with clustered noise while preserving color profile.
            An array representing the noisy image with clustered noise added, maintaining the original color profile.
    """
    # Load the input image
    image = cv2.imread(image_to_change_path)

    # Create a copy of the original image to work with
    noisy_image = image.copy()

    # Generate random parameters for the elliptical noise cluster
    center_x = random_generator.randint(0, image.shape[1])  # Random x-coordinate for cluster center
    center_y = random_generator.randint(0, image.shape[0])  # Random y-coordinate for cluster center
    major_axis = random_generator.randint(min_ellipse_length, max_ellipse_length)  # Random major axis length
    minor_axis = random_generator.randint(min_ellipse_width, max_ellipse_width)  # Random minor axis length
    angle = random_generator.randint(0, 180)  # Random rotation angle for the cluster

    # Create an initial mask filled with zeros
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Draw an ellipse on the mask based on the generated parameters
    cv2.ellipse(mask, (center_x, center_y), (major_axis, minor_axis), angle, 0, 360, 255, -1)

    # Find the coordinates of non-zero pixels in the mask
    non_zero_pixels = np.column_stack(np.where(mask))

    # Generate a random number of changes to apply within the specified range
    max_changes = random_generator.randint(min_changes, max_changes)
    applied_changes = 0  # Counter for tracking the number of changes applied

    # Loop over non-zero pixels and apply changes until the maximum number of changes is reached
    for row, col in non_zero_pixels:
        if applied_changes >= max_changes:
            break  # Exit the loop once the maximum number of changes has been reached

        # Clip pixel coordinates to image boundaries
        new_row = np.clip(row, 0, noisy_image.shape[0] - 1)
        new_col = np.clip(col, 0, noisy_image.shape[1] - 1)

        # Generate a random intensity change for each color channel
        intensity_change = random_generator.randint(min_intensity, max_intensity, size=3)

        # Add the intensity change to the pixel color while clipping values to the [0, 255] range
        noisy_color = np.clip(noisy_image[new_row, new_col] + intensity_change, 0, 255)

        # Apply the new color to the noisy image
        noisy_image[new_row, new_col] = noisy_color

        # Increment the counter for applied changes
        applied_changes += 1

    # Return the modified noisy image
    return noisy_image


def add_clustered_noise_random_ellipse_to_array_mut(image, random_generator, num_clusters,
                                                    min_cluster_density, max_cluster_density, min_intensity,
                                                    max_intensity,
                                                    min_pix_neighborhood, max_pix_neighborhood, min_ellipse_length,
                                                    max_ellipse_length, min_ellipse_width, max_ellipse_width):
    """
    Adds clustered elliptical noise to an image, based on the original pixel value.

    Args:
        image_to_change_path (str): Path to the input image.
            The path to the image that will NOT have clustered noise added to it (base image).

        random_generator (np.random.Generator): Random number generator.
            An instance of NumPy's random number generator used for consistent randomness across all calls in this
            project for experiment running purposes.

        num_clusters (int): Number of clusters of noise to add.
            The total number of noise clusters to be added to the image.

        min_cluster_density (int): Minimum cluster density.
            The lower bound for the number of pixels in a cluster that will be modified with noise.

        max_cluster_density (int): Maximum cluster density.
            The upper bound for the number of pixels in a cluster that will be modified with noise.

        min_intensity (int): Minimum intensity change.
            The lower bound for the intensity change applied to each pixel within the noise clusters.

        max_intensity (int): Maximum intensity change.
            The upper bound for the intensity change applied to each pixel within the noise clusters.

        min_pix_neighborhood (int): Minimum pixel neighborhood offset.
            The lower bound for the offset used to select neighboring pixels for applying noise.

        max_pix_neighborhood (int): Maximum pixel neighborhood offset.
            The upper bound for the offset used to select neighboring pixels for applying noise.

        min_ellipse_length (int): Minimum length of the major axis of the elliptical cluster.
            The lower bound for the length of the major axis of the elliptical noise clusters.

        max_ellipse_length (int): Maximum length of the major axis of the elliptical cluster.
            The upper bound for the length of the major axis of the elliptical noise clusters.

        min_ellipse_width (int): Minimum width of the elliptical cluster.
            The lower bound for the width of the elliptical noise clusters, controlling their narrowness.

        max_ellipse_width (int): Maximum width of the elliptical cluster.
            The upper bound for the width of the elliptical noise clusters, controlling their width variation.

    Returns:
        numpy.ndarray: Noisy image with clustered noise while preserving color profile.
            An array representing the noisy image with clustered noise added, maintaining the original color profile.
    """
    # image = cv2.imread(image)

    # Initialize the noisy image with a copy of the original image
    noisy_image = image.copy()

    # Apply noise by slightly changing colors of random non-black points within the cluster
    for _ in range(num_clusters):
        center_x = random_generator.randint(0, image.shape[1])
        center_y = random_generator.randint(0, image.shape[0])
        # Adjusts length of the major axis of the elliptical cluster.
        major_axis = random_generator.randint(min_ellipse_length, max_ellipse_length)
        # Controls how narrow the  cluster is along its shortest dimension
        minor_axis = random_generator.randint(min_ellipse_width, max_ellipse_width)
        angle = random_generator.randint(0, 180)  # Define the rotation of the elliptical cluster

        mask = np.zeros(image.shape[:2], dtype=np.uint8)  # initial mask filled with zeros, subsequent code will modify
        cv2.ellipse(mask, (center_x, center_y), (major_axis, minor_axis), angle, 0, 360, 255, -1)

        non_zero_pixels = np.column_stack(np.where(mask))
        for row, col in non_zero_pixels:
            pixel = noisy_image[row, col]
            if not np.all(pixel == 0):  # Skip black pixels
                # Increase this range for denser clusters
                num_pixels_to_change = random_generator.randint(min_cluster_density,
                                                                max_cluster_density)
                for _ in range(num_pixels_to_change):
                    # Offsets are used to select neighboring pixels for applying noise.
                    # Adjust this range for pixel neighborhood
                    rand_row = random_generator.randint(min_pix_neighborhood,
                                                        max_pix_neighborhood)
                    # Adjust this range for pixel neighborhood
                    rand_col = random_generator.randint(min_pix_neighborhood,
                                                        max_pix_neighborhood)
                    new_row = np.clip(row + rand_row, 0, noisy_image.shape[0] - 1)
                    new_col = np.clip(col + rand_col, 0, noisy_image.shape[1] - 1)
                    noisy_color = np.clip(noisy_image[new_row, new_col] + random_generator.randint(min_intensity,
                                                                                                   max_intensity,
                                                                                                   size=3), 0, 255)
                    noisy_image[new_row, new_col] = noisy_color

    return noisy_image  # Return the noisy image as a numpy array


def add_clustered_noise_random_ellipse_fast_to_array_mut(image, random_generator, min_intensity,
                                                         max_intensity, min_ellipse_length, max_ellipse_length,
                                                         min_ellipse_width, max_ellipse_width, min_changes,
                                                         max_changes):
    """
    Adds clustered elliptical noise to an image, based on the original pixel value.

    Args:
        image_to_change_path (str): Path to the input image.
            The path to the image that will NOT have clustered noise added to it (base image).

        random_generator (np.random.Generator): Random number generator.
            An instance of NumPy's random number generator used for consistent randomness across all calls in this
            project for experiment running purposes.

        num_clusters (int): Number of clusters of noise to add.
            The total number of noise clusters to be added to the image.

        min_cluster_density (int): Minimum cluster density.
            The lower bound for the number of pixels in a cluster that will be modified with noise.

        max_cluster_density (int): Maximum cluster density.
            The upper bound for the number of pixels in a cluster that will be modified with noise.

        min_intensity (int): Minimum intensity change.
            The lower bound for the intensity change applied to each pixel within the noise clusters.

        max_intensity (int): Maximum intensity change.
            The upper bound for the intensity change applied to each pixel within the noise clusters.

        min_pix_neighborhood (int): Minimum pixel neighborhood offset.
            The lower bound for the offset used to select neighboring pixels for applying noise.

        max_pix_neighborhood (int): Maximum pixel neighborhood offset.
            The upper bound for the offset used to select neighboring pixels for applying noise.

        min_ellipse_length (int): Minimum length of the major axis of the elliptical cluster.
            The lower bound for the length of the major axis of the elliptical noise clusters.

        max_ellipse_length (int): Maximum length of the major axis of the elliptical cluster.
            The upper bound for the length of the major axis of the elliptical noise clusters.

        min_ellipse_width (int): Minimum width of the elliptical cluster.
            The lower bound for the width of the elliptical noise clusters, controlling their narrowness.

        max_ellipse_width (int): Maximum width of the elliptical cluster.
            The upper bound for the width of the elliptical noise clusters, controlling their width variation.

    Returns:
        numpy.ndarray: Noisy image with clustered noise while preserving color profile.
            An array representing the noisy image with clustered noise added, maintaining the original color profile.
    """
    # image = cv2.imread(image)
    # Create a copy of the original image to work with
    noisy_image = image.copy()
    # cv2.imshow("Before Mutation", image)

    # Generate random parameters for the elliptical noise cluster
    center_x = random_generator.randint(0, image.shape[1])  # Random x-coordinate for cluster center
    center_y = random_generator.randint(0, image.shape[0])  # Random y-coordinate for cluster center
    major_axis = random_generator.randint(min_ellipse_length, max_ellipse_length)  # Random major axis length
    minor_axis = random_generator.randint(min_ellipse_width, max_ellipse_width)  # Random minor axis length
    angle = random_generator.randint(0, 180)  # Random rotation angle for the cluster

    # Create an initial mask filled with zeros
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Draw an ellipse on the mask based on the generated parameters
    cv2.ellipse(mask, (center_x, center_y), (major_axis, minor_axis), angle, 0, 360, 255, -1)

    # Find the coordinates of non-zero pixels in the mask
    non_zero_pixels = np.column_stack(np.where(mask))

    # Generate a random number of changes to apply within the specified range
    max_changes = random_generator.randint(min_changes, max_changes)
    applied_changes = 0  # Counter for tracking the number of changes applied

    # Loop over non-zero pixels and apply changes until the maximum number of changes is reached
    for row, col in non_zero_pixels:
        if applied_changes >= max_changes:
            break  # Exit the loop once the maximum number of changes has been reached

        # Clip pixel coordinates to image boundaries
        new_row = np.clip(row, 0, noisy_image.shape[0] - 1)
        new_col = np.clip(col, 0, noisy_image.shape[1] - 1)

        # Generate a random intensity change for each color channel
        intensity_change = random_generator.randint(min_intensity, max_intensity, size=3)

        # Add the intensity change to the pixel color while clipping values to the [0, 255] range
        noisy_color = np.clip(noisy_image[new_row, new_col] + intensity_change, 0, 255)

        # Apply the new color to the noisy image
        noisy_image[new_row, new_col] = noisy_color

        # Increment the counter for applied changes
        applied_changes += 1

    # cv2.imshow("After Mutation", noisy_image)
    # cv2.waitKey(0)
    # Return the modified noisy image
    return noisy_image


def visual_difference_4x4_plot(original_path, best_evolved_image_path, images_saved_by_gen, output_image_path):
    """
    Create a 4x4 visualization of original, evolved, and highlighted difference images.

    Args:
        output_image_path (str): Where the image is going to be saved
        original_path (str): Path to the original image.
        best_evolved_image_path (str): Path to the noisy evolved image.
        images_saved_by_gen (list): List of generation numbers for saved images.

    Returns:
        None
    """

    # Load the original image and noisy image
    original_image = cv2.imread(original_path)
    best_evolved_image = cv2.imread(best_evolved_image_path)

    # Calculate the absolute pixel-wise difference between the images
    difference = cv2.absdiff(original_image, best_evolved_image)

    # Apply binary thresholding to create a mask of changed pixels
    threshold = 30  # Adjust this threshold based on your needs
    _, mask = cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY)

    # Find the indices of the changed pixels
    changed_indices = np.where(mask > 0)

    # Define the highlight color (e.g., red)
    highlight_color = (0, 0, 255)  # BGR format

    # Create a copy of the original image to work with
    highlighted_image = original_image.copy()

    # Update the pixel values of the changed pixels with the highlight color
    for row, col in zip(changed_indices[0], changed_indices[1]):
        highlighted_image[row, col] = highlight_color

    # Create an image with highlighted parts on a black background
    highlighted_on_black = np.zeros_like(original_image, dtype=np.uint8)
    highlighted_on_black[changed_indices[0], changed_indices[1]] = highlight_color

    # Plot shape
    max_columns = 4
    max_rows = 3

    # Display the original, evolved, and highlighted images
    fig, axes = plt.subplots(max_rows, max_columns, figsize=(13, 10))

    # Plot the original, noisy evolved, and differences on the first row
    axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(cv2.cvtColor(best_evolved_image, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Evolved Best Image")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title("Highlighted Differences Image")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(cv2.cvtColor(highlighted_on_black, cv2.COLOR_BGR2RGB))
    axes[0, 3].set_title("Highlighted Differences on Black")
    axes[0, 3].axis("off")

    # Generate the rest of the images
    i = 0
    for x in range(1, max_rows):
        for y in range(max_columns):
            gen = images_saved_by_gen[i]
            evolved_image_path = output_image_path + "_gen_" + str(gen) + "_evolved.jpg"
            evolved_image = cv2.imread(evolved_image_path)
            axes[x, y].imshow(cv2.cvtColor(evolved_image, cv2.COLOR_BGR2RGB))
            axes[x, y].set_title(f"Best at gen: {gen}")
            axes[x, y].axis("off")
            i += 1

    plt.tight_layout()
    plt.show()
