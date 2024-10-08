import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import random
import copy
import warnings

# For hyperparameter tuning
from sklearn.model_selection import ParameterGrid

# For data augmentation
from scipy.ndimage import gaussian_filter1d

# For model interpretability (SHAP values)
# import shap

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


# Define the LSTMClassifier
# Long Short-Term Memory (LSTM) classifier is a deep learning architecture that
# uses a type of recurrent neural network (RNN) to classify sequential data. Basically, time series data.
#
# This neural network model uses LSTM layers to handle sequential data. The model consists of:
# - An LSTM layer that processes the input sequences.
# - A fully connected layer (fc) that maps the LSTM output to class scores.
# - The forward method defines how the input data flows through the network.
#
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


# Function to load CSV files from a given path
def load_csv_files(path):
    csv_files = list(path.glob("*.csv"))
    dfs = [pd.read_csv(file) for file in csv_files]
    return dfs


# Function to compute features while handling missing data:
#
# - Sorting and Interpolating Data: Frames are sorted to maintain sequence order,
#   and missing frames are interpolated to ensure continuity.
#
# - Pivoting the DataFrame: The data is reshaped so that each frame is a row and
#   each landmark measurement is a column.
#
# - Handling Missing Angles: Missing angle values are filled using interpolation to
#   maintain data integrity.
#
# - Ensuring Consistent Features: The DataFrame is reindexed to include all expected
#   features, filling any missing ones with zeros.
#
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


# Data augmentation function to augment a sequence
#
# - Time Warping: Alters the speed of the sequence to simulate variations in movement speed.
#
# - Adding Noise: Introduces slight random variations to make the model robust against minor inconsistencies.
#
# - Scaling: Adjusts the magnitude of the data to simulate different sizes or strengths of movements.
#
def augment_sequence(sequence, seq_len):
    # Time warping: randomly adjust the speed of the sequence
    warp_factor = np.random.uniform(0.8, 1.2)
    indices = np.round(np.linspace(0, len(sequence) - 1, seq_len)).astype(int)
    indices = np.clip(indices, 0, len(sequence) - 1)
    sequence = sequence[indices]
    # Add random noise to the sequence
    noise = np.random.normal(0, 0.01, sequence.shape)
    sequence += noise
    # Scale the sequence
    scale_factor = np.random.uniform(0.9, 1.1)
    sequence *= scale_factor
    return sequence


# Prepare data by computing features and adding labels
def prepare_data_for_model(df, label):
    # Compute features for the DataFrame
    features_df = compute_features_with_resilient_landmarks(df)
    # Convert labels to integers
    features_df["label"] = 0 if label == "normal" else 1
    return features_df


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


# Prepare sequences for the model, with optional data augmentation
# Feature Scaling: Normalizes the features using a StandardScaler to
# ensure that all features contribute equally to the model.
def prepare_sequences_for_model(dfs, seq_len=70, augment=False):
    sequences = []
    labels = []
    for df in dfs:
        # Normalize features using the global scaler
        features = df.drop(["frame", "label"], axis=1)
        features_scaled = pd.DataFrame(
            global_scaler.transform(features), columns=features.columns
        )
        features_scaled["label"] = df["label"].values
        features_scaled["frame"] = df["frame"].values
        # Create sequences from the scaled features
        seqs, lbls = create_sequences(features_scaled, seq_len)
        if augment:
            # Apply data augmentation to each sequence
            seqs = [augment_sequence(seq, seq_len) for seq in seqs]
        sequences.extend(seqs)
        labels.extend(lbls)
    return np.array(sequences), np.array(labels)


# Model file name
model_name = "parkinsons_classifier_model_complete.pth"
model_path = Path(model_name)

if not model_path.exists():
    # The model does not exist, proceed to train it

    # Load datasets
    normal_dfs = load_csv_files(normal_path)
    parkinsons_dfs = load_csv_files(parkinsons_path)

    # Combine all DataFrames for scaling
    all_dfs = normal_dfs + parkinsons_dfs

    # Process all DataFrames
    normal_processed_dfs = [prepare_data_for_model(df, "normal") for df in normal_dfs]
    parkinsons_processed_dfs = [
        prepare_data_for_model(df, "parkinsons") for df in parkinsons_dfs
    ]
    all_processed_dfs = normal_processed_dfs + parkinsons_processed_dfs

    # Combine all features to fit the scaler
    all_features = pd.concat(
        [df.drop(["frame", "label"], axis=1) for df in all_processed_dfs],
        ignore_index=True,
    )

    # Fit the global scaler
    global_scaler = StandardScaler()
    global_scaler.fit(all_features)

    # Implement k-fold cross-validation
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Hyperparameter tuning grid
    param_grid = {
        "hidden_size": [64, 128],
        "num_layers": [1, 2],
        "learning_rate": [0.001, 0.0005],
        "batch_size": [32, 64],
    }

    # Variables to store best model parameters
    best_accuracy = 0
    best_params = None
    best_model_state = None

    # Combine data and labels
    all_dfs_combined = normal_dfs + parkinsons_dfs
    all_labels = ["normal"] * len(normal_dfs) + ["parkinsons"] * len(parkinsons_dfs)
    all_dfs_with_labels = list(zip(all_dfs_combined, all_labels))

    # Grid search over hyperparameters
    for params in ParameterGrid(param_grid):
        fold_accuracies = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(all_dfs_with_labels)):
            # Split DataFrames into training and validation sets
            train_data = [all_dfs_with_labels[i] for i in train_idx]
            val_data = [all_dfs_with_labels[i] for i in val_idx]

            train_dfs = [data[0] for data in train_data]
            train_labels = [data[1] for data in train_data]
            val_dfs = [data[0] for data in val_data]
            val_labels = [data[1] for data in val_data]

            # Prepare data
            train_processed_dfs = [
                prepare_data_for_model(df, label)
                for df, label in zip(train_dfs, train_labels)
            ]
            val_processed_dfs = [
                prepare_data_for_model(df, label)
                for df, label in zip(val_dfs, val_labels)
            ]

            # Reindex DataFrames to have consistent features
            for i in range(len(train_processed_dfs)):
                train_processed_dfs[i] = train_processed_dfs[i].reindex(
                    columns=["frame"] + feature_columns + ["label"], fill_value=0
                )
            for i in range(len(val_processed_dfs)):
                val_processed_dfs[i] = val_processed_dfs[i].reindex(
                    columns=["frame"] + feature_columns + ["label"], fill_value=0
                )

            # Prepare sequences
            seq_len = 70
            X_train, y_train = prepare_sequences_for_model(
                train_processed_dfs, seq_len=seq_len, augment=True
            )
            X_val, y_val = prepare_sequences_for_model(
                val_processed_dfs, seq_len=seq_len
            )

            if len(X_train) == 0 or len(X_val) == 0:
                continue  # Skip if no sequences are available

            # Convert to tensors
            X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
            y_train = torch.tensor(y_train, dtype=torch.long).to(device)
            X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
            y_val = torch.tensor(y_val, dtype=torch.long).to(device)

            # Create datasets and dataloaders
            train_ds = TensorDataset(X_train, y_train)
            val_ds = TensorDataset(X_val, y_val)
            train_dl = DataLoader(
                train_ds, batch_size=params["batch_size"], shuffle=True
            )
            val_dl = DataLoader(val_ds, batch_size=params["batch_size"], shuffle=False)

            # Define the model with current hyperparameters
            input_size = X_train.shape[2]
            hidden_size = params["hidden_size"]
            num_layers = params["num_layers"]
            num_classes = 2

            model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(
                device
            )

            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

            # Training loop
            num_epochs = 5  # Reduced for hyperparameter tuning
            for epoch in range(num_epochs):
                model.train()
                for sequences, labels in train_dl:
                    sequences = sequences.to(device)
                    labels = labels.to(device)
                    outputs = model(sequences)
                    loss = criterion(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # Evaluate on validation set
            model.eval()
            with torch.no_grad():
                outputs = model(X_val)
                _, predicted = torch.max(outputs.data, 1)
                val_accuracy = accuracy_score(y_val.cpu(), predicted.cpu())
                fold_accuracies.append(val_accuracy)

        if len(fold_accuracies) == 0:
            continue  # Skip if no folds were processed

        avg_accuracy = np.mean(fold_accuracies)
        print(f"Params: {params}, Average Validation Accuracy: {avg_accuracy:.4f}")

        # Update best model if current model has better accuracy
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_params = params
            best_model_state = copy.deepcopy(model.state_dict())

    print(f"Best Params: {best_params}, Best Validation Accuracy: {best_accuracy:.4f}")

    # Retrain the model on the full dataset with best parameters
    train_processed_dfs = normal_processed_dfs + parkinsons_processed_dfs

    # Reindex DataFrames to have consistent features
    for i in range(len(train_processed_dfs)):
        train_processed_dfs[i] = train_processed_dfs[i].reindex(
            columns=["frame"] + feature_columns + ["label"], fill_value=0
        )

    # Prepare sequences
    X_train, y_train = prepare_sequences_for_model(
        train_processed_dfs, seq_len=seq_len, augment=True
    )

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)

    # Create dataset and dataloader
    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=best_params["batch_size"], shuffle=True)

    # Define the model with best parameters
    input_size = X_train.shape[2]
    hidden_size = best_params["hidden_size"]
    num_layers = best_params["num_layers"]
    num_classes = 2

    model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(best_model_state)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["learning_rate"])

    # Training loop with increased epochs for final training
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for sequences, labels in train_dl:
            sequences = sequences.to(device)
            labels = labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_dl)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "parkinsons_classifier_model.pth")

    # Save the checkpoint with model state and parameters
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_classes": num_classes,
        "global_scaler": global_scaler,  # Save the scaler
    }
    torch.save(checkpoint, model_name)


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
