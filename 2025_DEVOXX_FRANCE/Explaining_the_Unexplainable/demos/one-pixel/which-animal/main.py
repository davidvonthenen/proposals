# Importing necessary libraries
import os
import random
import warnings

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Disable future and user warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

# Device setup: Prioritize MPS, then CUDA, and then CPU
device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon) for training.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA (GPU) for training.")
else:
    print("Using CPU for training.")

# Set dataset paths and model path
train_dir = "training_set"
test_dir = "test_set"
model_path = "cat_dog_classifier.pth"

# Define image transformations without resizing since images are already 224x224
train_transform = transforms.Compose([
    transforms.TrivialAugmentWide(),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.ToTensor()
])

# Create a safe dataset by subclassing ImageFolder to catch errors
from torchvision.datasets import ImageFolder

class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except Exception as e:
            print(f"Error processing file {self.samples[index][0]}: {e}")
            raise

# Load datasets using SafeImageFolder
train_data = SafeImageFolder(train_dir, transform=train_transform)
test_data = SafeImageFolder(test_dir, transform=test_transform)

# Create DataLoaders with num_workers set to 0 for debugging purposes
BATCH_SIZE = 32
NUM_WORKERS = 0  # Use 0 to help reveal full exceptions in the main process

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

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
        # The classifier assumes inputs are 224x224.
        self.classifier = nn.Sequential(
            nn.Flatten()
        )
        # Adjust the fully connected layer to output three classes: cat and dog.
        # The flattened dimension is based on the output size from the convolution layers.
        self.fc = nn.Linear(512 * 28 * 28, 2)

    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.classifier(x)
        x = self.fc(x)
        return x

# Early stopping training process
if not os.path.exists(model_path):
    print("Training model...")
    model = ImageClassifier().to(device)

    def train_step(model, dataloader, loss_fn, optimizer):
        model.train()
        train_loss, train_acc = 0, 0
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item() / len(y_pred)
        return train_loss / len(dataloader), train_acc / len(dataloader)

    def test_step(model, dataloader, loss_fn):
        model.eval()
        test_loss, test_acc = 0, 0
        with torch.inference_mode():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                loss = loss_fn(y_pred, y)
                test_loss += loss.item()
                y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
                test_acc += (y_pred_class == y).sum().item() / len(y_pred)
        return test_loss / len(dataloader), test_acc / len(dataloader)

    def train(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs, patience=3):
        """
        Trains the model for a maximum number of epochs or exits early if test accuracy does not improve.
        :param model: The neural network model.
        :param train_dataloader: DataLoader for training.
        :param test_dataloader: DataLoader for testing/validation.
        :param optimizer: Optimizer for training.
        :param loss_fn: Loss function.
        :param epochs: Maximum number of epochs for training.
        :param patience: Number of epochs to wait for improvement before early stopping.
        :return: A dictionary with the training and testing metrics.
        """
        results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
        best_accuracy = 0  # Best observed test accuracy
        epochs_no_improve = 0  # Number of consecutive epochs without improvement

        for epoch in tqdm(range(epochs)):
            train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer)
            test_loss, test_acc = test_step(model, test_dataloader, loss_fn)

            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)

            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

            # Early stopping logic
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                epochs_no_improve = 0  # Reset counter when improvement is found
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"No improvement for {patience} consecutive epochs. Early stopping triggered.")
                break

        return results

    NUM_EPOCHS = 10
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_results = train(model, train_dataloader, test_dataloader, optimizer, loss_fn, NUM_EPOCHS)

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# Load the model for inference
model = ImageClassifier().to(device)
model.load_state_dict(torch.load(model_path))
print("Model loaded successfully.")

# Predict function for a given image
def predict_image(image_path, model, transform, class_labels):
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

class_labels = ["cat", "dog"]

# Sample predictions for test images
test_image_path = "good-cat.jpg"
predicted_class, probabilities = predict_image(test_image_path, model, test_transform, class_labels)
print(f"Cat (Good/Normal) Image - Predicted class: {predicted_class}, Probabilities: {probabilities}")

test_image_path_2 = "dog.jpg"
predicted_class_2, probabilities_2 = predict_image(test_image_path_2, model, test_transform, class_labels)
print(f"Dog Image: Predicted class - {predicted_class_2}, Probabilities: {probabilities_2}")

test_image_path3 = "bad-cat.jpg"
predicted_class_3, probabilities_3 = predict_image(test_image_path3, model, test_transform, class_labels)
print(f"Cat (Bad/Pixel) Image - Predicted class: {predicted_class_3}, Probabilities: {probabilities_3}")
