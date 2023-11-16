import torch
import torch.nn as nn
from PIL import Image

from torchvision import transforms

# 5. Build the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 2)  # Two classes: 7 and 9

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Create the model instanceee
loaded_model = Net()

# Load the saved weights
loaded_model.load_state_dict(torch.load('mnist_7_9_classifier_model_with_aug.pth'))

# Set the model to evaluation mode
loaded_model.eval()


# Load the image using PIL
image_path = '../../images/test_images/base_7.png'  # Replace with your image path
image = Image.open(image_path).convert('L')  # Convert to grayscale

# Resize and normalize the image
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

image_tensor = transform(image).unsqueeze(0)  # Add a batch dimension

with torch.no_grad():
    prediction = loaded_model(image_tensor)
    predicted_class = prediction.argmax(dim=1).item()

    # Get the confidence of the prediction
    probabilities = torch.nn.functional.softmax(prediction, dim=1)
    confidence = probabilities[0][predicted_class].item()

# Convert back to the original labels (7 and 9) and print with confidence
if predicted_class == 0:
    print(f"The predicted digit is 7 with {confidence*100:.2f}% confidence.")
else:
    print(f"The predicted digit is 9 with {confidence*100:.2f}% confidence.")
