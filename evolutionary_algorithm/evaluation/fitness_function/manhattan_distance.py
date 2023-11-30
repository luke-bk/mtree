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

def classify_image_dcm(loaded_model, image_array):
    # Load and preprocess the image
    image_array = np.uint8(image_array)
    image = Image.fromarray(image_array).convert('RGB')  # Convert to RGB
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust to match your model's expected input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard ImageNet normalization
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add a batch dimension

    # Predict the class
    with torch.no_grad():
        prediction = loaded_model(image_tensor)
        predicted_class = prediction.argmax(dim=1).item()
        probabilities = torch.nn.functional.softmax(prediction, dim=1)
        confidence = probabilities[0][predicted_class].item()

    # Return the predicted class and confidence
    return predicted_class, confidence

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
    adjusted_distance = distance * (1 - 0.9 * confidence)
    if class_name == current_class:
        adjusted_distance = 999999
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

    class_name, confidence = classify_image_dcm(loaded_model, Image.fromarray(np.uint8(image_one)))

    # Calculate the Manhattan distance
    distance = np.sum(np.abs(image_one - image_two))
    # adjusted_distance = distance * (1 - y) # The more confident, the better
    adjusted_distance = distance * (1 - 0.9 * confidence)
    if class_name == current_class:
        adjusted_distance = 999999
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