import sys
import random
import warnings
import numpy as np
import pandas as pd
import torch
from torch import nn
from pathlib import Path

# -----------------------------------------------------------------------------
#                            Configuration & Setup
# -----------------------------------------------------------------------------
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

# Prefer Apple Silicon MPS, then CUDA, then CPU
device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")

# Fix seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# The saved checkpoint file
CHECKPOINT_NAME = "parkinsons_classifier_model_complete.pth"
SEQ_LEN = 70

# -----------------------------------------------------------------------------
#                        Model Definition (Exact Same)
# -----------------------------------------------------------------------------
class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier for sequential data.
    A fully connected layer is applied on the final time step's output.
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# -----------------------------------------------------------------------------
#               Feature Computation (Exact Same with +1e-8)
# -----------------------------------------------------------------------------
landmarks = [
    "LEFT_ANKLE", "LEFT_ELBOW", "LEFT_HEEL", "LEFT_HIP", "LEFT_KNEE",
    "LEFT_SHOULDER", "LEFT_WRIST", "RIGHT_ANKLE", "RIGHT_ELBOW",
    "RIGHT_HEEL", "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_SHOULDER", "RIGHT_WRIST",
]

feature_columns = []
for lm in landmarks:
    for axis in ["x", "y", "z"]:
        feature_columns.append(f"{axis}_{lm}")
        feature_columns.append(f"{axis}_{lm}_vel")
        feature_columns.append(f"{axis}_{lm}_acc")

angle_features = [
    "LEFT_LEG_ANGLE", "RIGHT_LEG_ANGLE", "LEFT_ARM_ANGLE", "RIGHT_ARM_ANGLE",
    "LEFT_BODY_ARM_ANGLE", "RIGHT_BODY_ARM_ANGLE", "STEP_LENGTH"
]
feature_columns.extend(angle_features)

def compute_features_with_resilient_landmarks(df):
    """
    Sorts frames, interpolates missing data, computes velocities, accelerations,
    angles, and step length. Reindexes to the expected feature columns.
    """
    df = df.sort_values("frame").reset_index(drop=True)
    df = df.infer_objects(copy=False)
    df = df.set_index("frame").interpolate(method="linear").reset_index()

    # Pivot to wide format
    df_pivot = df.pivot_table(index="frame", columns="landmark", values=["x", "y", "z"])
    df_pivot.columns = ["_".join(col).strip() for col in df_pivot.columns.values]
    available_landmarks = set(col.split("_")[1] for col in df_pivot.columns if "_" in col)

    # Velocity and acceleration
    for lm in available_landmarks:
        for axis in ["x", "y", "z"]:
            col_name = f"{axis}_{lm}"
            if col_name in df_pivot.columns:
                df_pivot[f"{col_name}_vel"] = df_pivot[col_name].diff().fillna(0)
                df_pivot[f"{col_name}_acc"] = df_pivot[f"{col_name}_vel"].diff().fillna(0)

    def calculate_angle(df_local, p1, p2, p3):
        v1 = (df_local[[f"x_{p1}", f"y_{p1}", f"z_{p1}"]].values -
              df_local[[f"x_{p2}", f"y_{p2}", f"z_{p2}"]].values)
        v2 = (df_local[[f"x_{p3}", f"y_{p3}", f"z_{p3}"]].values -
              df_local[[f"x_{p2}", f"y_{p2}", f"z_{p2}"]].values)
        dot_product = np.einsum("ij,ij->i", v1, v2)
        norm_v1 = np.linalg.norm(v1, axis=1)
        norm_v2 = np.linalg.norm(v2, axis=1)
        # NOTE the +1e-8 to avoid zero division
        cos_angle = dot_product / (norm_v1 * norm_v2 + 1e-8)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)

    # Angles and step length (only if the landmarks are present)
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

    df_pivot = df_pivot.interpolate(method="linear").bfill().ffill()
    df_pivot = df_pivot.reindex(columns=feature_columns, fill_value=0)

    return df_pivot.reset_index()

def create_sequences(df, seq_len):
    """
    Slides a window of length seq_len over the DataFrame (row-wise).
    Returns (list_of_np_arrays, list_of_labels).
    Label is from last frame in each window if 'label' column is present.
    """
    sequences = []
    labels = []
    num_frames = len(df)
    if num_frames < seq_len:
        return sequences, labels

    for i in range(num_frames - seq_len + 1):
        df_slice = df.iloc[i : i + seq_len]
        # Drop columns not used in features
        cols_to_drop = ["frame"]
        if "label" in df_slice.columns:
            cols_to_drop.append("label")
        seq = df_slice.drop(columns=cols_to_drop, errors="ignore").values
        sequences.append(seq)
        # If there's a label column, use the label from the last row in the window
        if "label" in df_slice.columns:
            lbl = df_slice["label"].iloc[-1]
            labels.append(lbl)

    return sequences, labels

# -----------------------------------------------------------------------------
#              Inference Function: Predict on New CSV
# -----------------------------------------------------------------------------
def predict_on_new_csv(model, scaler, csv_path, seq_len=SEQ_LEN):
    """
    1) Loads the CSV and computes features.
    2) Scales them using the provided scaler.
    3) Creates sequences of length seq_len.
    4) Runs the model on all sequences and averages the output probabilities.
    Returns: (predicted_label, confidence).
    """
    if not Path(csv_path).exists():
        print(f"CSV path does not exist: {csv_path}")
        return (None, 0.0)

    # Load data
    new_df = pd.read_csv(csv_path)
    features_df = compute_features_with_resilient_landmarks(new_df)
    # Reindex for consistent columns
    features_df = features_df.reindex(columns=["frame"] + feature_columns, fill_value=0)

    # Scale
    temp_features = features_df.drop(columns=["frame"], errors="ignore")
    scaled_array = scaler.transform(temp_features)  # shape: (num_frames, feature_dim)
    scaled_df = pd.DataFrame(scaled_array, columns=temp_features.columns)
    scaled_df["frame"] = features_df["frame"].values

    # Create sequences
    seqs, _ = create_sequences(scaled_df, seq_len)
    if len(seqs) == 0:
        print(f"No sequences of length {seq_len} found in {csv_path}.")
        return (None, 0.0)

    # Convert to Torch and predict
    X_new = torch.tensor(seqs, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(X_new)
        probs = torch.softmax(outputs, dim=1)  # shape: (num_seq, 2)
        avg_probs = torch.mean(probs, dim=0)   # shape: (2,)
        predicted_label_idx = torch.argmax(avg_probs).item()
        label_str = "normal" if predicted_label_idx == 0 else "parkinsons"
        confidence = avg_probs[predicted_label_idx].item()

    return (label_str, confidence)

# -----------------------------------------------------------------------------
#                           Main Execution
# -----------------------------------------------------------------------------
def main():
    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_NAME, map_location=device)
    input_size = checkpoint["input_size"]
    hidden_size = checkpoint["hidden_size"]
    num_layers = checkpoint["num_layers"]
    num_classes = checkpoint["num_classes"]
    global_scaler = checkpoint["global_scaler"]

    # Reconstruct model and load state
    model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Run inference
    csv_path = "./test/normal.csv"
    label_str, confidence = predict_on_new_csv(model, global_scaler, csv_path, seq_len=SEQ_LEN)
    print(f"File: {csv_path}, Predicted: {label_str}, Confidence: {confidence:.4f}")

    csv_path = "./test/parkinsons.csv"
    label_str, confidence = predict_on_new_csv(model, global_scaler, csv_path, seq_len=SEQ_LEN)
    print(f"File: {csv_path}, Predicted: {label_str}, Confidence: {confidence:.4f}")

if __name__ == "__main__":
    main()
