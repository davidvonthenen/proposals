import pandas as pd
from pathlib import Path
import numpy as np
import torch
from torch import nn
import random
import warnings

# Disable warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

# Define paths to the data directories
path = Path("tabular")
normal_path = path / "normal"
parkinsons_path = path / "parkinsons"

# Set device to GPU if available, otherwise fall back to CPU
# Prioritize MPS (Apple Silicon), then CUDA (NVIDIA GPUs), then CPU
device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon) for training.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA (GPU) for training.")
else:
    print("Using CPU for training.")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# Define the LSTMClassifier Classifier
# This neural network model uses LSTM layers to handle sequential data. The model consists of:
# - An LSTM layer that processes the input sequences.
# - A fully connected layer (fc) that maps the LSTM output to class scores.
# - The forward method defines how the input data flows through the network.
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Define LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Define a fully connected output layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Forward propagate through LSTM
        out, _ = self.lstm(x, (h0, c0))
        # Get the output from the last time step
        out = self.fc(out[:, -1, :])
        return out


# List of all possible landmarks
landmarks = [
    "LEFT_ANKLE",
    "LEFT_ELBOW",
    "LEFT_HEEL",
    "LEFT_HIP",
    "LEFT_KNEE",
    "LEFT_SHOULDER",
    "LEFT_WRIST",
    "RIGHT_ANKLE",
    "RIGHT_ELBOW",
    "RIGHT_HEEL",
    "RIGHT_HIP",
    "RIGHT_KNEE",
    "RIGHT_SHOULDER",
    "RIGHT_WRIST",
]

# Generate all feature column names
feature_columns = []
for landmark in landmarks:
    for axis in ["x", "y", "z"]:
        feature_columns.append(f"{axis}_{landmark}")
        feature_columns.append(f"{axis}_{landmark}_vel")
        feature_columns.append(f"{axis}_{landmark}_acc")

# Add angle features to the feature columns
angle_features = [
    "LEFT_LEG_ANGLE",
    "RIGHT_LEG_ANGLE",
    "LEFT_ARM_ANGLE",
    "RIGHT_ARM_ANGLE",
    "LEFT_BODY_ARM_ANGLE",
    "RIGHT_BODY_ARM_ANGLE",
    "STEP_LENGTH",
]
feature_columns.extend(angle_features)


# Function to compute features while handling missing data
def compute_features_with_resilient_landmarks(df):
    # Ensure frame numbers are sorted
    df = df.sort_values("frame").reset_index(drop=True)

    # Convert columns to numeric types to prevent warnings
    df = df.infer_objects(copy=False)

    # Interpolate missing frames based on the 'frame' column
    df = df.set_index("frame").interpolate(method="linear").reset_index()

    # Pivot the DataFrame to have frames as rows and landmarks as columns
    df_pivot = df.pivot_table(index="frame", columns="landmark", values=["x", "y", "z"])

    # Flatten multi-level column names
    df_pivot.columns = ["_".join(col).strip() for col in df_pivot.columns.values]

    # Identify available landmarks in the data
    available_landmarks = set(
        [col.split("_")[1] for col in df_pivot.columns if "_" in col]
    )

    # For each landmark, compute velocity and acceleration
    for landmark in available_landmarks:
        for axis in ["x", "y", "z"]:
            col_name = f"{axis}_{landmark}"
            if col_name in df_pivot.columns:
                # Compute velocity as difference between frames
                df_pivot[f"{col_name}_vel"] = df_pivot[col_name].diff().fillna(0)
                # Compute acceleration as difference of velocities
                df_pivot[f"{col_name}_acc"] = (
                    df_pivot[f"{col_name}_vel"].diff().fillna(0)
                )

    # Function to calculate angles between three points
    def calculate_angle(df, p1, p2, p3):
        # Vectors between points
        v1 = (
            df[[f"x_{p1}", f"y_{p1}", f"z_{p1}"]].values
            - df[[f"x_{p2}", f"y_{p2}", f"z_{p2}"]].values
        )
        v2 = (
            df[[f"x_{p3}", f"y_{p3}", f"z_{p3}"]].values
            - df[[f"x_{p2}", f"y_{p2}", f"z_{p2}"]].values
        )
        # Compute angles using dot product formula
        dot_product = np.einsum("ij,ij->i", v1, v2)
        norm_v1 = np.linalg.norm(v1, axis=1)
        norm_v2 = np.linalg.norm(v2, axis=1)
        cos_angle = dot_product / (norm_v1 * norm_v2)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)

    # Calculate angles between specified joints if data is available
    # Leg angles
    if all(
        [
            f"x_{landmark}" in df_pivot.columns
            for landmark in ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"]
        ]
    ):
        df_pivot["LEFT_LEG_ANGLE"] = calculate_angle(
            df_pivot, "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"
        )
    if all(
        [
            f"x_{landmark}" in df_pivot.columns
            for landmark in ["RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"]
        ]
    ):
        df_pivot["RIGHT_LEG_ANGLE"] = calculate_angle(
            df_pivot, "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"
        )

    # Arm angles
    if all(
        [
            f"x_{landmark}" in df_pivot.columns
            for landmark in ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"]
        ]
    ):
        df_pivot["RIGHT_ARM_ANGLE"] = calculate_angle(
            df_pivot, "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"
        )
    if all(
        [
            f"x_{landmark}" in df_pivot.columns
            for landmark in ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"]
        ]
    ):
        df_pivot["LEFT_ARM_ANGLE"] = calculate_angle(
            df_pivot, "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"
        )

    # Body-arm angles
    if all(
        [
            f"x_{landmark}" in df_pivot.columns
            for landmark in ["RIGHT_ELBOW", "RIGHT_SHOULDER", "RIGHT_HIP"]
        ]
    ):
        df_pivot["RIGHT_BODY_ARM_ANGLE"] = calculate_angle(
            df_pivot, "RIGHT_ELBOW", "RIGHT_SHOULDER", "RIGHT_HIP"
        )
    if all(
        [
            f"x_{landmark}" in df_pivot.columns
            for landmark in ["LEFT_ELBOW", "LEFT_SHOULDER", "LEFT_HIP"]
        ]
    ):
        df_pivot["LEFT_BODY_ARM_ANGLE"] = calculate_angle(
            df_pivot, "LEFT_ELBOW", "LEFT_SHOULDER", "LEFT_HIP"
        )

    # Step length calculation
    if all(
        [
            f"x_{landmark}" in df_pivot.columns
            for landmark in ["LEFT_ANKLE", "RIGHT_ANKLE"]
        ]
    ):
        df_pivot["STEP_LENGTH"] = np.sqrt(
            (df_pivot["x_LEFT_ANKLE"] - df_pivot["x_RIGHT_ANKLE"]) ** 2
            + (df_pivot["y_LEFT_ANKLE"] - df_pivot["y_RIGHT_ANKLE"]) ** 2
            + (df_pivot["z_LEFT_ANKLE"] - df_pivot["z_RIGHT_ANKLE"]) ** 2
        )

    # Handle missing data by interpolating missing values
    df_pivot = df_pivot.interpolate(method="linear").bfill().ffill()

    # Ensure DataFrame has all expected feature columns
    df_pivot = df_pivot.reindex(columns=feature_columns, fill_value=0)

    return df_pivot.reset_index()


# Function to create sequences from the data
def create_sequences(df, seq_len):
    sequences = []
    labels = []
    num_frames = len(df)
    # Skip if the number of frames is less than the sequence length
    if num_frames < seq_len:
        return sequences, labels
    for i in range(num_frames - seq_len + 1):
        df_slice = df.iloc[i : i + seq_len]
        # Drop 'frame' and 'label' columns if they exist
        cols_to_drop = ["frame"]
        if "label" in df_slice.columns:
            cols_to_drop.append("label")
        # Get sequence data
        seq = df_slice.drop(cols_to_drop, axis=1).values
        sequences.append(seq)
        # Get the label from the last frame in the sequence
        if "label" in df_slice.columns:
            label = df_slice["label"].iloc[-1]
            labels.append(label)
    return sequences, labels


# Model file name
model_name = "parkinsons_classifier_model_complete.pth"
model_path = Path(model_name)


# Function to preprocess and predict on a new CSV file
def predict_on_new_csv(model, csv_path, seq_len=70):
    model.eval()
    new_df = pd.read_csv(csv_path)
    features_df = compute_features_with_resilient_landmarks(new_df)
    # Reindex to ensure consistent features
    features_df = features_df.reindex(columns=["frame"] + feature_columns, fill_value=0)
    # Scale features using the global scaler
    features = features_df.drop(["frame"], axis=1)
    features_scaled = pd.DataFrame(
        global_scaler.transform(features), columns=features.columns
    )
    features_scaled["frame"] = features_df["frame"].values
    # Create sequences
    seqs, _ = create_sequences(features_scaled, seq_len)
    if len(seqs) == 0:
        print(f"No sequences of length {seq_len} in {csv_path}")
        return
    sequences = torch.tensor(seqs, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(sequences)
        probs = torch.softmax(outputs, dim=1)
        avg_probs = torch.mean(probs, dim=0)
        predicted_label = torch.argmax(avg_probs).item()
        label = "normal" if predicted_label == 0 else "parkinsons"
        print(
            f"Prediction for {csv_path}: {label}, Probability: {avg_probs[predicted_label]:.4f}"
        )


# Load the checkpoint
checkpoint = torch.load(model_name)

# Extract model parameters
input_size = checkpoint["input_size"]
hidden_size = checkpoint["hidden_size"]
num_layers = checkpoint["num_layers"]
num_classes = checkpoint["num_classes"]
global_scaler = checkpoint["global_scaler"]  # Load the scaler

# Reconstruct the model
model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
# Load the saved state dictionary
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Example prediction with a new CSV file
new_csv_path = "test/normal.csv"  # Replace with your new CSV path
predict_on_new_csv(model, new_csv_path)

new_csv_path = "test/parkinsons.csv"  # Replace with your new CSV path
predict_on_new_csv(model, new_csv_path)
