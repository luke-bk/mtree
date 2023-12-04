from torch.utils.data import Dataset
import os
import pydicom

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn

import numpy as np
from PIL import Image

class DICOM_Dataset(Dataset):
    def __init__(self, image_ids, image_labels, image_base_dir, channels=3):
        self.image_ids = image_ids
        self.image_labels = image_labels
        self.image_base_dir = image_base_dir
        self.channels = channels

    def __getitem__(self, i):
        image_id = self.image_ids[i]
        label = self.image_labels[i]
        img_path = os.path.join(self.image_base_dir, image_id)

        if image_id[-4:] == '.dcm':
            dcm_image = pydicom.read_file(img_path)
            image = dcm_image.pixel_array
        else:
            img = Image.open(img_path)
            image = np.array(img)
            if len(image.shape) == 3:
                image = image[:, :, :0]  # Take only the first three channels

        # This suppresses potential division errors that may occur.
        # The resulting nan errors are dealt with in the subsequent line.
        # This could cause problems in the future, but I can't think of any at this time.
        with np.errstate(divide='ignore', invalid='ignore'):
            image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = np.nan_to_num(image, nan=0.0, posinf=1.0)

        if self.channels == 3:
            image = np.stack((image,)*3, axis=0)

        # image = image.astype('int16') # remove this when you are normalizing the images again.
        image = torch.from_numpy(image).to(dtype=torch.float32)

        return {'image': image, 'label': label}

    def __len__(self):
        return len(self.image_ids)


# Build the model
# VGG16#
class VGG_net(nn.Module):

    def __init__(self, in_channels=3, num_classes=1,
                 LAYERS=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512,
                         'M']):  # Defaults to VGG16. Change layers to adjust.
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.conv_layers = self.create_conv_layers(LAYERS)
        self.fcs = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes))

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        for x in architecture:
            if type(x) == int:
                out_channels = x
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x


# LOADS THE MODEL FOR INFERENCE##
weights = r'../../model/dfo_vgg16_weights.pt'
loaded_model = VGG_net()  # No need to specify device here

# Update this line
x = torch.load(weights, map_location=torch.device('cpu'))

loaded_model.load_state_dict(x)
loaded_model.eval()  # Continue to use eval mode for inference
loaded_model.train(False)

# Suppose you have the following single image:
single_image_path = '../../../images/dfo_images/dfo_class_1.dcm'  # or 'path/to/your/image.png'
dummy_label = 1  # Just a placeholder, since we don't need the label for inference

# Step 1: Create the dataset instance
# Note: You need to ensure that 'image_ids' and 'image_labels' are lists, even for one image
dataset = DICOM_Dataset(
    image_ids=[os.path.basename(single_image_path)],
    image_labels=[dummy_label],
    image_base_dir=os.path.dirname(single_image_path)
)

# Step 2: Get the preprocessed image tensor using the __getitem__ method
image_data = dataset.__getitem__(0)  # '0' because it's the only image
image_tensor = image_data['image'].unsqueeze(0)  # Add batch dimension

# Step 3: Move the image tensor to the CPU (since the model is on the CPU)
image_tensor = image_tensor.to('cpu')

# Step 4: Pass the image tensor through the model to get the raw output
with torch.no_grad():
    output = loaded_model(image_tensor)

# Step 5: Apply softmax if your model outputs logits
probabilities = torch.nn.functional.softmax(output, dim=1)
predicted_class = probabilities.argmax(dim=1).item()
confidence = probabilities[0][predicted_class].item()

# Step 6: Print the classification result
print(f'Predicted class: {predicted_class}, Confidence: {confidence:.4f}')

import os
from torch.utils.data import DataLoader
import torch.nn.functional as F

def process_folder(folder_path, model, batch_size=60):
    image_ids = [img for img in os.listdir(folder_path) if img.endswith('.dcm')]
    image_labels = [0] * len(image_ids)  # Dummy labels

    dataset = DICOM_Dataset(image_ids=image_ids, image_labels=image_labels, image_base_dir=folder_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Keep track of the current image index across batches
    current_image_index = 0

    # Process each batch
    for batch in data_loader:
        image_tensors = batch['image']
        image_tensors = image_tensors.to('cpu')

        with torch.no_grad():
            outputs = model(image_tensors)
            # Use sigmoid for binary classification
            probabilities = torch.sigmoid(outputs).squeeze()  # Squeeze to remove any extra dimensions

        # Print or store the results
        for i in range(probabilities.size(0)):
            image_id = image_ids[current_image_index]
            probability = probabilities[i].item()  # Probability of being in the positive class
            predicted_class = 1 if probability >= 0.5 else 0  # Class prediction based on threshold
            print(f'Image: {image_id}, Predicted class: {predicted_class}, Probability: {probability:.4f}')
            current_image_index += 1

# Usage
folder_path = '../../../images/test_dfo_sample/'  # Update with your folder path
process_folder(folder_path, loaded_model)