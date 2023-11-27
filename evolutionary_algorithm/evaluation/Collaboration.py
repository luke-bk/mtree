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


def combine_quads(quads):
    """
    Recursively combine quads to form a larger image.

    Args:
        quads: A list of quads to be combined.

    Returns:
        The combined image from the quads.
    """
    if len(quads) == 1:
        return quads[0]
    elif len(quads) == 4:
        top_half = np.hstack((quads[0], quads[3]))
        bottom_half = np.hstack((quads[1], quads[2]))
        return np.vstack((top_half, bottom_half))
    else:
        # Handle cases where the number of quads is not 4
        # This part needs custom logic based on how you want to combine them
        pass


def collaborate_image_new(collaboration, generation):
    """
    Collaborate with other populations to create a complete solution which is an image.
    Groups chromosomes by parent names and then combines their images.

    Args:
        collaboration: The collaboration list containing chromosomes.

    Returns:
        grouped_images: A dictionary of combined images grouped by parent names.
    """
    # Grouping chromosomes by parent names
    grouped_chromosomes = {}
    sub_quad_keys = set()  # Use a set to avoid duplicate keys
    for collaborator in collaboration:
        parent_name = collaborator.parent_name
        if parent_name not in grouped_chromosomes:
            grouped_chromosomes[parent_name] = []
            if parent_name != '0':
                sub_quad_keys.add(parent_name)
        grouped_chromosomes[parent_name].append(collaborator.chromosome)

    # print (sub_quad_keys)
    # Start the reassembling process
    for key in sub_quad_keys:
        if key in grouped_chromosomes:
            # Combine its quadrants if the key exists
            # if len(sub_quad_keys) == 10:
            #     print (key)
            temp_image = combine_quads(grouped_chromosomes[key])
            # Determine the parent key by removing the last character from the child key
            parent_key = key[:-1]
            # Determine the position in the parent's list (last character of the key)
            position = int(key[-1])
            # Insert the combined image back into the parent at the correct position
            grouped_chromosomes[parent_key].insert(position, temp_image)

    # After processing all sub-quads, combine the main quadrants
    full_image = combine_quads(grouped_chromosomes['0'])  # Assuming '0' is the root node

    if generation == 104:

        new_width = 600
        new_height = 600
        resized_image = cv2.resize(full_image, (new_width, new_height))

        cv2.imshow('Image', resized_image)
        cv2.waitKey(0)  # Waits indefinitely for a key press
        cv2.destroyAllWindows()  # Closes the window after a key is pressed

    return full_image


def get_child_keys(parent_key, grouped_chromosomes):
    child_key_length = len(parent_key) + 1  # One more digit than the parent key
    return [key for key in grouped_chromosomes if key.startswith(parent_key) and len(key) == child_key_length]


def combine_quads_recursively(parent_key, grouped_chromosomes):
    """
    Recursively combines quads starting from the given parent key.

    Args:
        parent_key: The key of the parent quad.
        grouped_chromosomes: Dictionary of all quads grouped by keys.

    Returns:
        The combined image of the quads starting from the parent key.
    """
    if parent_key not in grouped_chromosomes:
        return None

    child_keys = get_child_keys(parent_key, grouped_chromosomes)
    child_images = []

    # Check if there are child quadrants
    if child_keys:
        for key in child_keys:
            child_image = combine_quads_recursively(key, grouped_chromosomes)
            if child_image is not None:
                child_images.append(child_image)

        if child_images:
            return combine_quads(child_images)
    else:
        # Directly combine the images under the current parent key
        return combine_quads(grouped_chromosomes[parent_key]) if parent_key in grouped_chromosomes else None


def collaborate_image_recursive(collaboration):
    """
    Collaborate to create a complete solution which is an image, from a series of arrays.

    Args:
        collaboration: The collaboration list containing chromosomes.

    Returns:
        The reassembled full image.
    """
    if len(collaboration) == 1:
        return collaboration[0].chromosome

    # Grouping chromosomes by parent names
    grouped_chromosomes = {}
    for collaborator in collaboration:
        parent_name = collaborator.parent_name
        if parent_name not in grouped_chromosomes:
            grouped_chromosomes[parent_name] = []
        grouped_chromosomes[parent_name].append(collaborator.chromosome)

    # Start the reassembling process from the root
    full_image = combine_quads_recursively('0', grouped_chromosomes)

    return full_image


def dynamic_image_combination(images, num_collaborators):
    """
    Combine images dynamically based on the number of collaborators.
    """
    # Implement logic based on how the images should be combined
    # This can be complex depending on the structure and number of images
    pass