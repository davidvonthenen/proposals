import warnings
import os

import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend to allow PNG saving
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path

# Captum
from captum.attr import IntegratedGradients, GradientShap, LRP, NoiseTunnel
from captum.attr import visualization as viz

from matplotlib.colors import LinearSegmentedColormap

# Disable future warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

# Device setup to prioritize MPS, then CUDA, and then CPU
device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon) for training.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA (GPU) for training.")
else:
    print("Using CPU for training.")

# Set paths
model_path = "cat_dog_classifier.pth"

# Define image transformations
IMAGE_SIZE = (224, 224)
train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])

# Define the CNN model
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )
        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(64, 512, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2)
        )
        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(512 * 28 * 28, 2)

    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = torch.flatten(x, 1)  # Replace Flatten layer
        x = self.fc(x)
        return x


# Load the model
model = ImageClassifier().to(device)
model.load_state_dict(torch.load(model_path))
print("Model loaded successfully.")

class_labels = ["cat", "dog"]

# Custom colormap for attribution visualization
default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)

def visualize_attributions_with_noise_tunnel(model, image_path, target_label):
    """
    Generate and save attributions visualization using Noise Tunnel with Integrated Gradients.

    Args:
        model (nn.Module): The trained model.
        image_path (str): Path to the input image file.
        target_label (int): The target label index for attribution.
    """
    model.eval()
    
    # Define image transformations
    IMAGE_SIZE = (224, 224)
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    transformed_image = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])(image)  # Unnormalized image for visualization
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Initialize Integrated Gradients and Noise Tunnel
    integrated_gradients = IntegratedGradients(model)
    noise_tunnel = NoiseTunnel(integrated_gradients)
    
    # Compute attributions
    attributions_ig_nt = noise_tunnel.attribute(
        input_tensor, nt_samples=10, nt_type='smoothgrad_sq', target=target_label
    )
    
    # Visualize and save attributions
    fig, ax = plt.subplots(figsize=(8, 8))
    _ = viz.visualize_image_attr_multiple(
        np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(transformed_image.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        ["original_image", "heat_map"],
        ["all", "positive"],
        cmap=default_cmap,
        show_colorbar=True
    )

    # Save the visualization to a file
    file_base = Path(image_path).stem  # Remove file extension from the image path
    output_path = f"{file_base}_noise.png"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    plt.close(fig)

def visualize_attributions_with_lrp(model, image_path, target_label):
    """
    Generate and save attributions visualization using Layer-Wise Relevance Propagation (LRP).

    Args:
        model (nn.Module): The trained model.
        image_path (str): Path to the input image file.
        target_label (int): The target label index for attribution.
    """
    model.eval()

    # Define image transformations
    IMAGE_SIZE = (224, 224)
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    transformed_image = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])(image)  # Unnormalized image for visualization
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Initialize LRP
    lrp = LRP(model)

    # Compute attributions
    attributions_lrp = lrp.attribute(input_tensor, target=target_label)

    # Visualize and save attributions
    fig, ax = plt.subplots(figsize=(8, 8))
    _ = viz.visualize_image_attr_multiple(
        np.transpose(attributions_lrp.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(transformed_image.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        ["original_image", "heat_map"],
        ["all", "positive"],
        cmap=default_cmap,
        show_colorbar=True,
        outlier_perc=2
    )

    # Save the visualization to a file
    file_base = Path(image_path).stem  # Remove file extension from the image path
    output_path = f"{file_base}_lrp.png"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    plt.close(fig)

def predict_and_visualize(image_path, model, transform, class_labels):
    """
    Predict the class of the image and generate attributions visualizations.

    Args:
        image_path (str): Path to the image to predict and analyze.
        model (nn.Module): Trained model for predictions.
        transform (torchvision.transforms.Compose): Transformations for input image.
        class_labels (list): List of class labels (e.g., ["cat", "dog"]).
    """
    model.eval()
    
    # Load and transform the image
    image = Image.open(image_path).convert("RGB")
    image_transformed = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.inference_mode():
        output = model(image_transformed)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        readable_probabilities = {class_labels[i]: f"{probabilities[0][i].item() * 100:.2f}%" for i in range(len(class_labels))}
        print(f"Predicted class: {class_labels[predicted_class]}, Probabilities: {readable_probabilities}")
    
    # Initialize attribution methods
    integrated_gradients = IntegratedGradients(model)
    noise_tunnel = NoiseTunnel(integrated_gradients)
    
    # Compute attributions using Integrated Gradients with Noise Tunnel
    attributions = noise_tunnel.attribute(image_transformed, nt_samples=10, nt_type='smoothgrad_sq', target=predicted_class)
    
    # Check for zero attributions
    if torch.all(attributions == 0):
        print("Warning: Attributions are all zeros. Skipping visualization.")
        return

    # Save positive and negative attributions
    file_base = Path(image_path).stem  # Remove file extension from the image path
    positive_attr_path = f"{file_base}_positive_attr.png"
    negative_attr_path = f"{file_base}_negative_attr.png"
    
    # Generate and save positive attribution visualization
    fig, ax = plt.subplots()
    try:
        viz.visualize_image_attr(
            np.transpose(attributions.squeeze().cpu().detach().numpy(), (1, 2, 0)),
            np.transpose(transform(image).squeeze().cpu().detach().numpy(), (1, 2, 0)),
            method="heat_map",
            sign="positive",
            cmap=default_cmap,
            show_colorbar=True,
            outlier_perc=2,
            plt_fig_axis=(fig, ax)
        )
        plt.savefig(positive_attr_path)
        plt.close(fig)
        print(f"Saved positive attribution visualization: {positive_attr_path}")
    except Exception as e:
        print(f"Failed to generate positive attributions: {e}")
        plt.close(fig)

    # Generate and save negative attribution visualization
    fig, ax = plt.subplots()
    try:
        viz.visualize_image_attr(
            np.transpose(attributions.squeeze().cpu().detach().numpy(), (1, 2, 0)),
            np.transpose(transform(image).squeeze().cpu().detach().numpy(), (1, 2, 0)),
            method="heat_map",
            sign="negative",
            cmap=default_cmap,
            show_colorbar=True,
            outlier_perc=2,
            plt_fig_axis=(fig, ax)
        )
        plt.savefig(negative_attr_path)
        plt.close(fig)
        print(f"Saved negative attribution visualization: {negative_attr_path}")
    except Exception as e:
        print(f"Failed to generate negative attributions: {e}")
        plt.close(fig)

# Example usage
test_image_path = "dog.jpg"
predict_and_visualize(test_image_path, model, test_transform, class_labels)
visualize_attributions_with_noise_tunnel(model, test_image_path, 1)
visualize_attributions_with_lrp(model, test_image_path, 1)
