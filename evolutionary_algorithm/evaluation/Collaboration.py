import cv2
import numpy as np


def collaborate(random_generator, binary_tree, leaf_node):
    """
    Collaborate with other populations to create a complete solution.

    Args:
        binary_tree: The binary tree containing leaf nodes.
        leaf_node: The current leaf node for which collaboration is being done.

    Returns:
        complete_solution: The complete solution after collaboration.
        sub_solution_index: The index where the solution needs to be inserted.
    """
    complete_solution = []
    sub_solution_index = None

    for index, collaborator_node in enumerate(binary_tree.get_leaf([])):
        if collaborator_node.population is not leaf_node.population:
            if collaborator_node.population.elite is not None:
                # Add elites from other populations
                complete_solution.append(collaborator_node.population.elite.clone())
            else:
                # Return a random choice if the elite doesn't exist
                complete_solution.append(random_generator.choice(collaborator_node.population.chromosomes).clone())
        else:
            sub_solution_index = index  # Save the index where the solution needs to be inserted

    return complete_solution, sub_solution_index


def collaborate_image(collaboration):
    """
    Collaborate with other populations to create a complete solution with images.

    Args:
        collaboration: The collaboration.

    Returns:
        complete_solution: The complete solution after collaboration.
        sub_solution_index: The index where the solution needs to be inserted.
    """
    num_collaborators = len(collaboration)
    images = []

    for collaborator_node in collaboration:
        image = collaborator_node.chromosome
        images.append(image)

    # Assuming 4 collaborators and images are quadrants
    if num_collaborators == 4:
        top_half = np.hstack((images[0], images[3]))
        bottom_half = np.hstack((images[1], images[2]))
        full_image = np.vstack((top_half, bottom_half))

        # print ("incollab")
        #
        # new_width = 600
        # new_height = 600
        # resized_image = cv2.resize(full_image, (new_width, new_height))
        #
        # cv2.imshow('Image', resized_image)
        # cv2.waitKey(0)  # Waits indefinitely for a key press
        # cv2.destroyAllWindows()  # Closes the window after a key is pressed
    else:
        # Handle other cases, possibly using a more dynamic approach
        full_image = image

    return full_image


def collaborate_image_new(collaboration):
    """
    Collaborate with other populations to create a complete solution which is an image.
    Groups chromosomes by parent names and then combines their images.

    Args:
        collaboration: The collaboration list containing chromosomes.

    Returns:
        grouped_images: A dictionary of combined images grouped by parent names.
    """
    # collection of the groups
    grouped_chromosomes = {}

    # Grouping chromosomes by parent names
    for collaborator_node in collaboration:
        parent_name = collaborator_node.parent_name
        if parent_name not in grouped_chromosomes:
            grouped_chromosomes[parent_name] = []
        grouped_chromosomes[parent_name].append(collaborator_node.chromosome)

    combined_images = {}

    # Stacking images for each group
    for parent_name, images in grouped_chromosomes.items():
        num_images = len(images)

        # Assuming images are quadrants and there are 4 images per group
        if num_images > 1:
            top_half = np.hstack((images[0], images[3]))
            bottom_half = np.hstack((images[1], images[2]))
            full_image = np.vstack((top_half, bottom_half))
        else:
            # For a different number of images, you might need a different approach
            full_image = images[0]  # Default to the first image if not 4 images

        combined_images[parent_name] = full_image

    return full_image


def dynamic_image_combination(images, num_collaborators):
    """
    Combine images dynamically based on the number of collaborators.
    """
    # Implement logic based on how the images should be combined
    # This can be complex depending on the structure and number of images
    pass