import os
import time
import random
import copy
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.ndimage import gaussian_filter1d  # For possible smoothing or augmentation

# Suppress future and user warnings for cleaner output during training
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

###############################################################################
#                            Device and Random Seed                           #
###############################################################################
# Set device: prefers MPS for Apple Silicon, then CUDA, otherwise CPU
device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon) for training.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA (GPU) for training.")
else:
    print("Using CPU for training.")

# Fix random seeds for reproducibility across torch, numpy, and python's random module
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

###############################################################################
#                          Hyperparameters & Config                           #
###############################################################################
# Grid search over these hyperparameters
param_grid = {
    "hidden_size": [128],
    "num_layers": [2],
    "learning_rate": [0.001],
    "batch_size": [16]
}

max_epochs = 30  # Maximum epochs for final training loop
patience = 3     # Early stopping patience counter
seq_len = 70     # Sequence length for LSTM input
checkpoint_name = "parkinsons_classifier_model_complete.pth"  # File name for saving the model checkpoint

###############################################################################
#                              Model Definition                               #
###############################################################################
class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier for sequential data.
    A fully connected layer is applied on the final time step's output for classification.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM layer with batch_first=True so input/output tensors are (batch, seq, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Final fully connected layer for classification
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden state (h0) and cell state (c0) with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Pass input through LSTM layer
        out, _ = self.lstm(x, (h0, c0))
        # Use the output from the final time step for classification
        out = self.fc(out[:, -1, :])
        return out

###############################################################################
#                           Feature Computation                               #
###############################################################################
# List of landmarks used in feature computation
landmarks = [
    "LEFT_ANKLE", "LEFT_ELBOW", "LEFT_HEEL", "LEFT_HIP", "LEFT_KNEE",
    "LEFT_SHOULDER", "LEFT_WRIST", "RIGHT_ANKLE", "RIGHT_ELBOW",
    "RIGHT_HEEL", "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_SHOULDER", "RIGHT_WRIST",
]

# Generate feature column names for positions, velocities, and accelerations
feature_columns = []
for lm in landmarks:
    for axis in ["x", "y", "z"]:
        feature_columns.append(f"{axis}_{lm}")
        feature_columns.append(f"{axis}_{lm}_vel")
        feature_columns.append(f"{axis}_{lm}_acc")

# List of additional angle and step length features
angle_features = [
    "LEFT_LEG_ANGLE", "RIGHT_LEG_ANGLE",
    "LEFT_ARM_ANGLE", "RIGHT_ARM_ANGLE",
    "LEFT_BODY_ARM_ANGLE", "RIGHT_BODY_ARM_ANGLE",
    "STEP_LENGTH"
]
feature_columns.extend(angle_features)

def compute_features_with_resilient_landmarks(df):
    """
    Given a DataFrame of raw landmark data (frame, landmark, x, y, z),
    computes additional features (velocities, accelerations, angles, step length)
    while handling missing frames via interpolation.
    """
    df = df.sort_values("frame").reset_index(drop=True)
    df = df.infer_objects(copy=False)

    # Interpolate missing values across frames using linear interpolation
    df = df.set_index("frame").interpolate(method="linear").reset_index()

    # Pivot the DataFrame to wide format with each column corresponding to a specific feature for a landmark
    df_pivot = df.pivot_table(index="frame", columns="landmark", values=["x", "y", "z"])
    df_pivot.columns = ["_".join(col).strip() for col in df_pivot.columns.values]

    # Determine which landmarks are available based on column names
    available_landmarks = set(col.split("_")[1] for col in df_pivot.columns if "_" in col)

    # Compute velocity and acceleration (frame-to-frame differences) for each landmark and axis
    for lm in available_landmarks:
        for axis in ["x", "y", "z"]:
            col_name = f"{axis}_{lm}"
            if col_name in df_pivot.columns:
                df_pivot[f"{col_name}_vel"] = df_pivot[col_name].diff().fillna(0)
                df_pivot[f"{col_name}_acc"] = df_pivot[f"{col_name}_vel"].diff().fillna(0)

    def calculate_angle(df_local, p1, p2, p3):
        """Helper function to compute the angle at point p2 formed by (p1, p2, p3)."""
        v1 = (
            df_local[[f"x_{p1}", f"y_{p1}", f"z_{p1}"]].values -
            df_local[[f"x_{p2}", f"y_{p2}", f"z_{p2}"]].values
        )
        v2 = (
            df_local[[f"x_{p3}", f"y_{p3}", f"z_{p3}"]].values -
            df_local[[f"x_{p2}", f"y_{p2}", f"z_{p2}"]].values
        )
        dot_product = np.einsum("ij,ij->i", v1, v2)
        norm_v1 = np.linalg.norm(v1, axis=1)
        norm_v2 = np.linalg.norm(v2, axis=1)
        cos_angle = dot_product / (norm_v1 * norm_v2 + 1e-8)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)

    # Compute leg and arm angles, as well as body-arm angles and step length,
    # only if the required landmarks are available
    if {"LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"} <= available_landmarks:
        df_pivot["LEFT_LEG_ANGLE"] = calculate_angle(df_pivot, "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE")
    if {"RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"} <= available_landmarks:
        df_pivot["RIGHT_LEG_ANGLE"] = calculate_angle(df_pivot, "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE")
    if {"RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"} <= available_landmarks:
        df_pivot["RIGHT_ARM_ANGLE"] = calculate_angle(df_pivot, "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST")
    if {"LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"} <= available_landmarks:
        df_pivot["LEFT_ARM_ANGLE"] = calculate_angle(df_pivot, "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST")
    if {"RIGHT_ELBOW", "RIGHT_SHOULDER", "RIGHT_HIP"} <= available_landmarks:
        df_pivot["RIGHT_BODY_ARM_ANGLE"] = calculate_angle(df_pivot, "RIGHT_ELBOW", "RIGHT_SHOULDER", "RIGHT_HIP")
    if {"LEFT_ELBOW", "LEFT_SHOULDER", "LEFT_HIP"} <= available_landmarks:
        df_pivot["LEFT_BODY_ARM_ANGLE"] = calculate_angle(df_pivot, "LEFT_ELBOW", "LEFT_SHOULDER", "LEFT_HIP")
    if {"LEFT_ANKLE", "RIGHT_ANKLE"} <= available_landmarks:
        df_pivot["STEP_LENGTH"] = np.sqrt(
            (df_pivot["x_LEFT_ANKLE"] - df_pivot["x_RIGHT_ANKLE"]) ** 2 +
            (df_pivot["y_LEFT_ANKLE"] - df_pivot["y_RIGHT_ANKLE"]) ** 2 +
            (df_pivot["z_LEFT_ANKLE"] - df_pivot["z_RIGHT_ANKLE"]) ** 2
        )

    # Final interpolation to handle any remaining missing values and fill forward/backward
    df_pivot = df_pivot.interpolate(method="linear").bfill().ffill()
    # Ensure DataFrame has exactly the expected feature columns (fill missing columns with zeros)
    df_pivot = df_pivot.reindex(columns=feature_columns, fill_value=0)

    return df_pivot.reset_index()

def prepare_data_for_model(df, label):
    """
    Processes a raw CSV (loaded as a DataFrame), computes features, and sets
    the 'label' column to 0 (normal) or 1 (parkinsons).
    """
    features_df = compute_features_with_resilient_landmarks(df)
    features_df["label"] = 0 if label == "normal" else 1
    return features_df

def create_sequences(df, seq_len):
    """
    Slides a window of fixed length (seq_len) over the DataFrame to create sequences.
    Uses the label from the final frame in the sequence as the sequence label.
    """
    sequences = []
    labels = []
    num_frames = len(df)
    if num_frames < seq_len:
        return sequences, labels

    # Create overlapping sequences by sliding one frame at a time
    for i in range(num_frames - seq_len + 1):
        df_slice = df.iloc[i : i + seq_len]
        if "label" in df_slice.columns:
            lbl = df_slice["label"].iloc[-1]  # Use label of the last frame in the sequence
        else:
            lbl = None

        # Exclude non-feature columns from the sequence data
        seq = df_slice.drop(columns=["frame", "label"], errors="ignore").values
        sequences.append(seq)
        if lbl is not None:
            labels.append(lbl)

    return sequences, labels

def augment_sequence(sequence, seq_len):
    """
    Applies three augmentation techniques to a sequence:
      - Time warping: randomly stretches/compresses the sequence.
      - Adding random noise.
      - Scaling the sequence.
    Ensures the output sequence is exactly seq_len long.
    """
    # Time warp: change the sequence's time scale by a random factor
    warp_factor = np.random.uniform(0.8, 1.2)
    num_points = max(int(round(seq_len * warp_factor)), 1)
    # Create new indices for the warped sequence using linear spacing
    new_indices = np.round(np.linspace(0, len(sequence) - 1, num_points)).astype(int)
    new_indices = np.clip(new_indices, 0, len(sequence) - 1)
    warped_seq = sequence[new_indices]

    # Adjust sequence length to exactly seq_len by truncating or padding with the last value
    if len(warped_seq) > seq_len:
        warped_seq = warped_seq[:seq_len]
    elif len(warped_seq) < seq_len:
        pad_size = seq_len - len(warped_seq)
        warped_seq = np.vstack((warped_seq, np.repeat(warped_seq[-1:], pad_size, axis=0)))

    # Add random noise to simulate sensor variations
    noise = np.random.normal(0, 0.01, warped_seq.shape)
    warped_seq += noise

    # Scale sequence values by a random factor to simulate differences in movement intensity
    scale_factor = np.random.uniform(0.9, 1.1)
    warped_seq *= scale_factor

    return warped_seq

def prepare_sequences_for_model(dfs, seq_len=70, scaler=None, augment=False):
    """
    Converts each processed DataFrame in the list 'dfs' into sequences,
    applies scaling using the provided scaler, and optionally performs augmentation.
    Returns numpy arrays of sequences and their corresponding labels.
    """
    all_sequences = []
    all_labels = []
    for df in dfs:
        # Extract feature columns (excluding frame and label)
        features = df.drop(columns=["frame", "label"], errors="ignore")
        if scaler is not None:
            # Standardize features using the scaler fitted on training data
            scaled_values = scaler.transform(features)
            scaled_features = pd.DataFrame(scaled_values, columns=features.columns)
        else:
            scaled_features = features.copy()

        # Add back frame and label columns for sequence creation
        scaled_features["label"] = df["label"].values
        scaled_features["frame"] = df["frame"].values

        # Create sequences from the scaled DataFrame
        seqs, lbls = create_sequences(scaled_features, seq_len)
        if augment:
            # Augment each sequence if augmentation is enabled
            seqs = [augment_sequence(s, seq_len) for s in seqs]

        all_sequences.extend(seqs)
        all_labels.extend(lbls)

    return np.array(all_sequences), np.array(all_labels)

###############################################################################
#                          Inference/Check Logic                              #
###############################################################################
def check_inference_correctness(model, scaler):
    """
    Performs a quick check on two test CSV files (normal and parkinsons) located in a "test" folder
    to verify that the model predicts the correct class for each.
    """
    data_path = Path("data")

    # Gather test data
    test_normal_path = data_path / "test" / "Normal"
    test_parkinsons_path = data_path / "test" / "Parkinsons"
    test_normal_files = list(test_normal_path.glob("*.csv"))
    test_parkinsons_files = list(test_parkinsons_path.glob("*.csv"))

    # get a single file randomly from test in both the Normal and Parkinsons folder
    normal_csv = random.choice(test_normal_files)
    parkinsons_csv = random.choice(test_parkinsons_files)

    if not normal_csv.exists() or not parkinsons_csv.exists():
        print("No normal.csv or parkinsons.csv found in test/ directory for final check.")
        return True

    label_normal, _ = predict_on_new_csv(model, scaler, str(normal_csv))
    label_parkinsons, _ = predict_on_new_csv(model, scaler, str(parkinsons_csv))

    if label_normal == "normal" and label_parkinsons == "parkinsons":
        return True
    return False

def predict_on_new_csv(model, scaler, csv_path, seq_len=70):
    """
    Preprocesses a new CSV file (not seen during training) to compute features,
    scales them, creates sequences, and runs the model to obtain predictions.
    Returns the predicted label and average confidence.
    """
    model.eval()
    if not os.path.exists(csv_path):
        print(f"CSV path does not exist: {csv_path}")
        return None, 0.0

    new_df = pd.read_csv(csv_path)
    features_df = compute_features_with_resilient_landmarks(new_df)
    features_df = features_df.reindex(columns=["frame"] + feature_columns, fill_value=0)

    temp_features = features_df.drop(columns=["frame"], errors="ignore")
    scaled = scaler.transform(temp_features)
    scaled_df = pd.DataFrame(scaled, columns=temp_features.columns)
    scaled_df["frame"] = features_df["frame"].values

    seqs, _ = create_sequences(scaled_df, seq_len)
    if len(seqs) == 0:
        print(f"No sequences of length {seq_len} found in {csv_path}.")
        return None, 0.0

    X_new = torch.tensor(seqs, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(X_new)
        probs = torch.softmax(outputs, dim=1)
        avg_probs = torch.mean(probs, dim=0)
        predicted_label = torch.argmax(avg_probs).item()
        label_str = "normal" if predicted_label == 0 else "parkinsons"
        confidence = avg_probs[predicted_label].item()
        print(f"File: {csv_path}, Predicted: {label_str}, Confidence: {confidence:.4f}")
        return label_str, confidence

###############################################################################
#                               Training Logic                                #
###############################################################################
def train_model():
    """
    Performs a single training pass:
      - Loads training, validation, and test CSV files from separate directories.
      - Processes the CSVs into DataFrames with computed features.
      - Builds a scaler from the training data.
      - Performs a grid search over hyperparameters with a short training loop.
      - Selects the best hyperparameters based on validation accuracy.
      - Retrains on the combined train+validation set with early stopping.
      - Evaluates on the test set (if available) and saves a checkpoint.
      - Finally, verifies the model's predictions on test CSVs in the project root "test/" folder.
    """
    data_path = Path("data")

    # Gather train data
    train_normal_path = data_path / "train" / "Normal"
    train_parkinsons_path = data_path / "train" / "Parkinsons"
    train_normal_files = list(train_normal_path.glob("*.csv"))
    train_parkinsons_files = list(train_parkinsons_path.glob("*.csv"))
    train_files = train_normal_files + train_parkinsons_files
    train_labels = (["normal"] * len(train_normal_files)) + (["parkinsons"] * len(train_parkinsons_files))

    # Gather validation data
    val_normal_path = data_path / "validate" / "Normal"
    val_parkinsons_path = data_path / "validate" / "Parkinsons"
    val_normal_files = list(val_normal_path.glob("*.csv"))
    val_parkinsons_files = list(val_parkinsons_path.glob("*.csv"))
    val_files = val_normal_files + val_parkinsons_files
    val_labels = (["normal"] * len(val_normal_files)) + (["parkinsons"] * len(val_parkinsons_files))

    # Gather test data
    test_normal_path = data_path / "test" / "Normal"
    test_parkinsons_path = data_path / "test" / "Parkinsons"
    test_normal_files = list(test_normal_path.glob("*.csv"))
    test_parkinsons_files = list(test_parkinsons_path.glob("*.csv"))
    test_files = test_normal_files + test_parkinsons_files
    test_labels = (["normal"] * len(test_normal_files)) + (["parkinsons"] * len(test_parkinsons_files))

    # Process CSV files into DataFrames with computed features
    train_dfs = [prepare_data_for_model(pd.read_csv(fp), lbl) for fp, lbl in zip(train_files, train_labels)]
    val_dfs = [prepare_data_for_model(pd.read_csv(fp), lbl) for fp, lbl in zip(val_files, val_labels)]
    test_dfs = [prepare_data_for_model(pd.read_csv(fp), lbl) for fp, lbl in zip(test_files, test_labels)]

    # Show which files are used (optional tracking output)
    print("\nValidation Files Used:")
    for vf, vlbl in zip(val_files, val_labels):
        print(f"  {vf} (label={vlbl})")

    print("\nTest Files Used:")
    for tf, tlbl in zip(test_files, test_labels):
        print(f"  {tf} (label={tlbl})")
    print()

    if len(train_dfs) == 0 or len(val_dfs) == 0:
        print("Not enough train/validation data to proceed. Exiting.")
        return False

    # Build a StandardScaler based on all training features
    all_train_features = pd.concat([
        df.drop(columns=["frame", "label"], errors="ignore") for df in train_dfs
    ], ignore_index=True)
    global_scaler = StandardScaler().fit(all_train_features)

    # Hyperparameter search: grid search over param_grid with a short training loop
    best_accuracy = 0.0
    best_params = None
    best_model_state = None

    for params in ParameterGrid(param_grid):
        # Prepare training and validation sequences using current hyperparameters
        X_train, y_train = prepare_sequences_for_model(
            train_dfs, seq_len=seq_len, scaler=global_scaler, augment=True
        )
        X_val, y_val = prepare_sequences_for_model(
            val_dfs, seq_len=seq_len, scaler=global_scaler, augment=False
        )

        if len(X_train) == 0 or len(X_val) == 0:
            continue

        input_size = X_train.shape[2]
        model = LSTMClassifier(
            input_size=input_size,
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            num_classes=2
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

        # Short training loop (5 epochs) to measure validation accuracy
        for _ in range(5):
            model.train()
            X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
            y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
            batch_size = params["batch_size"]

            # Create DataLoader
            train_ds = TensorDataset(X_train_t, y_train_t)
            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

            for xb, yb in train_dl:
                optimizer.zero_grad()
                outputs = model(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()

        # Evaluate model on validation set
        model.eval()
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        with torch.no_grad():
            val_outputs = model(X_val_t)
            predicted = torch.argmax(val_outputs, dim=1)
            val_acc = accuracy_score(y_val, predicted.cpu().numpy())

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_params = params
            best_model_state = copy.deepcopy(model.state_dict())

    print(f"\nBest Params from search: {best_params}, Val Accuracy: {best_accuracy:.4f}")
    if not best_params:
        print("No best parameters found (possibly no valid sequences). Exiting.")
        return False

    # Retrain using best hyperparameters on the combined train+val sets with early stopping
    full_train_dfs = train_dfs + val_dfs
    all_trainval_features = pd.concat([
        df.drop(columns=["frame", "label"], errors="ignore") for df in full_train_dfs
    ], ignore_index=True)
    global_scaler = StandardScaler().fit(all_trainval_features)

    X_trainval, y_trainval = prepare_sequences_for_model(
        full_train_dfs, seq_len=seq_len, scaler=global_scaler, augment=True
    )
    if len(X_trainval) == 0:
        print("No sequences in train+val set. Exiting.")
        return False

    input_size = X_trainval.shape[2]
    final_model = LSTMClassifier(
        input_size=input_size,
        hidden_size=best_params["hidden_size"],
        num_layers=best_params["num_layers"],
        num_classes=2
    ).to(device)
    final_model.load_state_dict(best_model_state)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params["learning_rate"])

    # DataLoader for final training
    X_tv_t = torch.tensor(X_trainval, dtype=torch.float32).to(device)
    y_tv_t = torch.tensor(y_trainval, dtype=torch.long).to(device)
    trainval_ds = TensorDataset(X_tv_t, y_tv_t)
    trainval_dl = DataLoader(trainval_ds, batch_size=best_params["batch_size"], shuffle=True)

    best_final_loss = float("inf")
    patience_counter = 0

    for epoch in range(max_epochs):
        final_model.train()
        total_loss = 0.0
        for xb, yb in trainval_dl:
            optimizer.zero_grad()
            outputs = final_model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(trainval_dl)
        print(f"[Final Training] Epoch [{epoch+1}/{max_epochs}], Loss: {avg_loss:.4f}")

        # Early stopping
        if avg_loss < best_final_loss:
            best_final_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Stopping early at epoch {epoch+1} due to no improvement.")
                break

    # Evaluate the final model on the test set
    if len(test_dfs) > 0:
        X_test, y_test = prepare_sequences_for_model(
            test_dfs, seq_len=seq_len, scaler=global_scaler, augment=False
        )
        if len(X_test) > 0:
            final_model.eval()
            X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
            with torch.no_grad():
                outputs = final_model(X_test_t)
                predicted = torch.argmax(outputs, dim=1)
                test_acc = accuracy_score(y_test, predicted.cpu().numpy())
            print(f"Final Test Accuracy: {test_acc:.4f}")
        else:
            print("No sequences in final test set (insufficient frames).")
    else:
        print("No final test set found. Skipping final test evaluation.")

    # Save the checkpoint
    checkpoint = {
        "model_state_dict": final_model.state_dict(),
        "input_size": input_size,
        "hidden_size": best_params["hidden_size"],
        "num_layers": best_params["num_layers"],
        "num_classes": 2,
        "global_scaler": global_scaler,
    }
    torch.save(checkpoint, checkpoint_name)
    print("\nFinal model checkpoint saved as:", checkpoint_name)

    # Quick sanity check on test/normal.csv and test/parkinsons.csv
    is_correct = check_inference_correctness(final_model, global_scaler)
    if not is_correct:
        print("Warning: Inference on test/normal.csv or test/parkinsons.csv did not match expected labels.")

    return is_correct

###############################################################################
#                              Main Execution                                 #
###############################################################################
if __name__ == "__main__":
    checkpoint_path = Path(checkpoint_name)

    # If checkpoint exists, skip training and only run the inference check
    if checkpoint_path.exists():
        print("Checkpoint found. Loading model and checking inference correctness...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model = LSTMClassifier(
            checkpoint["input_size"],
            checkpoint["hidden_size"],
            checkpoint["num_layers"],
            checkpoint["num_classes"]
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        scaler = checkpoint["global_scaler"]

        print("\nVerifying final inference on test/normal.csv and test/parkinsons.csv...")
        check_inference_correctness(model, scaler)
    else:
        # Perform a single training pass
        print("No checkpoint found. Starting a single training pass...")
        train_model()
