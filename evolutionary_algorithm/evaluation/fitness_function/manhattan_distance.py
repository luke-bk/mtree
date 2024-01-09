import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from torchvision import transforms

def classify_image(loaded_model, image_array):
    # Load and preprocess the image
    image_array = np.uint8(image_array)
    image = Image.fromarray(image_array).convert('L')  # Convert to grayscale
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add a batch dimension

    # Predict the class
    with torch.no_grad():
        prediction = loaded_model(image_tensor)
        predicted_class = prediction.argmax(dim=1).item()
        probabilities = torch.nn.functional.softmax(prediction, dim=1)
        confidence = probabilities[0][predicted_class].item()

    # Map the predicted class to the original labels
    class_label = 7 if predicted_class == 0 else 9
    return class_label, confidence

# Custom transform function for min-max normalization
def min_max_normalize(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    normalized = (tensor - min_val) / (max_val - min_val)
    return normalized

def classify_image_dcm(loaded_model, image_array):
    # Load and preprocess the image
    # Normalize and preprocess the image array as per your existing logic
    with np.errstate(divide='ignore', invalid='ignore'):
        image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
    image_array = np.nan_to_num(image_array, nan=0.0, posinf=1.0)

    image_array = np.stack((image_array,) * 3, axis=0)

    # Convert to torch tensor
    image_tensor = torch.from_numpy(image_array).to(dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    # Move the tensor to the CPU
    image_tensor = image_tensor.to('cpu')

    # Pass the tensor through the model
    with torch.no_grad():
        output = loaded_model(image_tensor)

    # Use sigmoid for binary classification
    probability = torch.sigmoid(output).item()
    predicted_class = 1 if probability >= 0.5 else 0  # Class prediction based on threshold

    # Return the predicted class and probability
    return predicted_class, probability


def manhattan_distance_fitness(loaded_model, image_one, image_two, current_class):
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
        print (image_one.shape)
        print (image_two.shape)
        raise ValueError("Images must have the same dimensions.")

    class_name, confidence = classify_image(loaded_model, Image.fromarray(np.uint8(image_one)))

    # Calculate the Manhattan distance
    distance = np.sum(np.abs(image_one - image_two))
    # adjusted_distance = distance * (1 - y) # The more confident, the better
    # adjusted_distance = distance * (1 - 0.9 * confidence)
    adjusted_distance = distance
    if class_name == current_class:
        adjusted_distance = 999999
    return adjusted_distance


def manhattan_distance_fitness_dcm_compare(loaded_model, image_one, image_two, current_class):
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
        print (image_one.shape)
        print (image_two.shape)
        raise ValueError("Images must have the same dimensions.")

    # class_name, confidence = classify_image_dcm(loaded_model, Image.fromarray(np.uint8(image_one)))
    class_name, probability = classify_image_dcm(loaded_model, image_one)

    # Calculate the Manhattan distance
    distance = np.sum(np.abs(image_one - image_two))
    # Adjust distance based on how far the probability is from the threshold
    # This will scale the adjustment factor based on the confidence of the classification
    # The adjustment factor will be smaller if the probability is close to 0.5 (uncertain classification)
    # and larger if the probability is close to 0.0 or 1.0 (certain classification)
    adjustment_factor = 1 - abs(probability - 0.5) * 2  # Scales the difference to range from 0 to 1
    adjusted_distance = distance * adjustment_factor
    # print(f"class name {class_name} and probability {probability}, with a fitness of {adjusted_distance}")
    return adjusted_distance

def manhattan_distance_fitness_dcm(loaded_model, image_one, image_two, current_class):
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
        print (image_one.shape)
        print (image_two.shape)
        raise ValueError("Images must have the same dimensions.")

    # class_name, confidence = classify_image_dcm(loaded_model, Image.fromarray(np.uint8(image_one)))
    class_name, probability = classify_image_dcm(loaded_model, image_one)

    # Calculate the Manhattan distance
    distance = np.sum(np.abs(image_one - image_two))

    if class_name == current_class:
        distance = 999999999
    # print(f"class name {class_name} and probability {probability}, with a fitness of {adjusted_distance}")
    return distance


def manhattan_distance_fitness_dcm_weighted(loaded_model, image_one, image_two, current_class):
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
        print (image_one.shape)
        print (image_two.shape)
        raise ValueError("Images must have the same dimensions.")

    # class_name, confidence = classify_image_dcm(loaded_model, Image.fromarray(np.uint8(image_one)))
    class_name, probability = classify_image_dcm(loaded_model, image_one)

    # Calculate the Manhattan distance
    distance = np.sum(np.abs(image_one - image_two))
    # Adjust distance based on how far the probability is from the threshold
    # This will scale the adjustment factor based on the confidence of the classification
    # The adjustment factor will be smaller if the probability is close to 0.5 (uncertain classification)
    # and larger if the probability is close to 0.0 or 1.0 (certain classification)
    adjustment_factor = abs(probability - 0.5) * 1.05  # Scales the difference to range from 0 to 1
    adjusted_distance = distance * adjustment_factor
    if class_name == current_class:
        adjusted_distance = 999999999
    # print(f"class name {class_name} and probability {probability}, with a fitness of {adjusted_distance}")
    return adjusted_distance


# def class_change_manhattan_distance_fitness(image_one, image_two, model):
#     """
#     Calculate the Manhattan distance between two images based on their RGB values,
#     adjusting the fitness score based on class similarity.
#
#     Args:
#         image_one (numpy.ndarray): First input image.
#         image_two (numpy.ndarray): Second input image.
#         model (torch.nn.Module): Neural network model for class prediction.
#
#     Returns:
#         float: Calculated fitness score.
#     """
#
#     # Ensure both images have the same dimensions
#     if image_one.shape != image_two.shape:
#         raise ValueError("Images must have the same dimensions.")
#
#     # Function to predict the class of an image using the model
#     def predict_class(image):
#         # Convert numpy image to PIL image, apply transformation and add batch dimension
#         image = Image.fromarray(image).convert('L')
#         image_tensor = transforms(image).unsqueeze(0)
#
#         # Predict class
#         with torch.no_grad():
#             prediction = model(image_tensor)
#             predicted_class = prediction.argmax(dim=1).item()
#         return predicted_class
#
#     # Predict classes for both images
#     class_one = predict_class(image_one)
#     class_two = predict_class(image_two)
#
#     # If both images belong to the same class, return a high fitness score
#     if class_one == class_two:
#         return float('inf')  # or some other very high number
#
#     # Otherwise, calculate the Manhattan distance
#     distance = np.sum(np.abs(image_one - image_two))
#     return distance