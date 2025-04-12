import os
import warnings
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Captum
from captum.attr import IntegratedGradients, LRP, NoiseTunnel
from captum.attr import visualization as viz
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend if running remotely
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

###############################################################################
#                              Device Setup                                   #
###############################################################################
device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon) for training.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA (GPU) for training.")
else:
    print("Using CPU for training.")

###############################################################################
#                           Model Definition                                  #
###############################################################################
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
        # If your checkpoint was trained for 2 classes (cat and dog)
        self.fc = nn.Linear(512 * 28 * 28, 2)

    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

###############################################################################
#                         Load the Pretrained Model                           #
###############################################################################
model_path = "cat_dog_classifier.pth"
model = ImageClassifier().to(device)
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()
print("Model loaded successfully.")

# Class labels for inference
class_labels = ["cat", "dog"]

###############################################################################
#                    Inference Function (from the new code)                   #
###############################################################################
def predict_image(image_path, model, transform, class_labels):
    """
    Predicts the class of the given image and returns both the predicted label
    and the probability of each class.
    """
    model.eval()
    with torch.inference_mode():
        image = Image.open(image_path).convert("RGB")
        image_transformed = transform(image).unsqueeze(0).to(device)
        output = model(image_transformed)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        readable_probabilities = {
            class_labels[i]: f"{probabilities[0][i].item() * 100:.2f}%"
            for i in range(len(class_labels))
        }
        return class_labels[predicted_class], readable_probabilities

###############################################################################
#               Captum Interpretability Functions (from old code)             #
###############################################################################
# Since all images are already 224x224, we remove any resizing operations.

# Transform for Captum: includes normalization but NO resizing
captum_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Unnormalized transform for visualization overlays (no resizing)
visual_transform = transforms.Compose([
    transforms.ToTensor()
])

default_cmap = LinearSegmentedColormap.from_list(
    'custom blue',
    [(0, '#ffffff'), (0.25, '#000000'), (1, '#000000')],
    N=256
)

def visualize_attributions_with_noise_tunnel(model, image_path, target_label):
    """
    Generate and save attributions visualization using Noise Tunnel with Integrated Gradients.
    """
    model.eval()
    image = Image.open(image_path).convert("RGB")

    # Unnormalized image for the background
    transformed_image = visual_transform(image)
    # Normalized tensor for model input
    input_tensor = captum_transform(image).unsqueeze(0).to(device)

    integrated_gradients = IntegratedGradients(model)
    noise_tunnel = NoiseTunnel(integrated_gradients)
    attributions_ig_nt = noise_tunnel.attribute(
        input_tensor,
        nt_samples=10,
        nt_type='smoothgrad_sq',
        target=target_label
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    viz.visualize_image_attr_multiple(
        np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(transformed_image.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        ["original_image", "heat_map"],
        ["all", "positive"],
        cmap=default_cmap,
        show_colorbar=True
    )
    file_base = Path(image_path).stem
    output_path = f"{file_base}_noise.png"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Noise Tunnel visualization saved to {output_path}")
    plt.close(fig)

def visualize_attributions_with_lrp(model, image_path, target_label):
    """
    Generate and save attributions visualization using Layer-Wise Relevance Propagation (LRP).
    """
    model.eval()
    image = Image.open(image_path).convert("RGB")

    # Unnormalized image for the background
    transformed_image = visual_transform(image)
    # Normalized tensor for model input
    input_tensor = captum_transform(image).unsqueeze(0).to(device)

    lrp = LRP(model)
    attributions_lrp = lrp.attribute(input_tensor, target=target_label)

    fig, ax = plt.subplots(figsize=(8, 8))
    viz.visualize_image_attr_multiple(
        np.transpose(attributions_lrp.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(transformed_image.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        ["original_image", "heat_map"],
        ["all", "positive"],
        cmap=default_cmap,
        show_colorbar=True,
        outlier_perc=2
    )
    file_base = Path(image_path).stem
    output_path = f"{file_base}_lrp.png"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"LRP visualization saved to {output_path}")
    plt.close(fig)

def predict_and_visualize(image_path, model, transform, class_labels):
    """
    Predict the class of the image and generate attributions visualizations 
    using Integrated Gradients (with Noise Tunnel) for both positive and negative.
    """
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image_transformed = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.inference_mode():
        output = model(image_transformed)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        readable_probabilities = {
            class_labels[i]: f"{probabilities[0][i].item() * 100:.2f}%"
            for i in range(len(class_labels))
        }
        print(f"Predicted class: {class_labels[predicted_class]}, Probabilities: {readable_probabilities}")

    integrated_gradients = IntegratedGradients(model)
    noise_tunnel = NoiseTunnel(integrated_gradients)
    attributions = noise_tunnel.attribute(
        image_transformed,
        nt_samples=10,
        nt_type='smoothgrad_sq',
        target=predicted_class
    )

    if torch.all(attributions == 0):
        print("Warning: Attributions are all zeros. Skipping visualization.")
        return

    file_base = Path(image_path).stem

    # Positive attributions
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
        positive_attr_path = f"{file_base}_positive_attr.png"
        plt.savefig(positive_attr_path)
        plt.close(fig)
        print(f"Saved positive attribution visualization: {positive_attr_path}")
    except Exception as e:
        print(f"Failed to generate positive attributions: {e}")
        plt.close(fig)

    # Negative attributions
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
        negative_attr_path = f"{file_base}_negative_attr.png"
        plt.savefig(negative_attr_path)
        plt.close(fig)
        print(f"Saved negative attribution visualization: {negative_attr_path}")
    except Exception as e:
        print(f"Failed to generate negative attributions: {e}")
        plt.close(fig)

###############################################################################
#                          Example Usage / Testing                             #
###############################################################################
# Images are assumed to be 224x224, so no resizing here.
test_transform = transforms.Compose([
    transforms.ToTensor()  # No resizing
])

# For good-cat.jpg
test_image_path_1 = "good-cat.jpg"
pred_class_1, probs_1 = predict_image(test_image_path_1, model, test_transform, class_labels)
print(f"[New Inference] {test_image_path_1} -> Predicted: {pred_class_1}, Probabilities: {probs_1}")

predict_and_visualize(test_image_path_1, model, captum_transform, class_labels)
visualize_attributions_with_noise_tunnel(model, test_image_path_1, target_label=1)
visualize_attributions_with_lrp(model, test_image_path_1, target_label=1)

# For bad-cat.jpg
test_image_path_2 = "bad-cat.jpg"
pred_class_2, probs_2 = predict_image(test_image_path_2, model, test_transform, class_labels)
print(f"[New Inference] {test_image_path_2} -> Predicted: {pred_class_2}, Probabilities: {probs_2}")

predict_and_visualize(test_image_path_2, model, captum_transform, class_labels)
visualize_attributions_with_noise_tunnel(model, test_image_path_2, target_label=1)
visualize_attributions_with_lrp(model, test_image_path_2, target_label=1)
