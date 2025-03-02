#!/usr/bin/env python3

import os
import warnings

import torch
import pandas as pd
import numpy as np
from torch import nn
import torch.nn.functional as Fnn

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

###############################################################################
# 1. CNN+LSTM Classifier Definition
#    Matches the updated training code with 2 MaxPool2d => /4 in time dimension.
###############################################################################
class CNNLSTMClassifier(nn.Module):
    def __init__(
        self,
        num_features=64,       # number of acoustic features per frame
        cnn_filters=32,
        hidden_size=128,
        num_layers=2,
        num_classes=2,
        dropout=0.5,
        vocab_size=100,        # distinct "word" tokens
        seq_tag_size=3,        # e.g. '-', 'START', 'CONTINUE'
        word_emb_dim=16,
        seq_emb_dim=4,
        bidirectional=True
    ):
        super().__init__()
        self.num_features = num_features
        self.cnn_filters = cnn_filters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.bidirectional = bidirectional

        # Three MaxPool2d => /8 in the time dimension
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=cnn_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=cnn_filters, out_channels=cnn_filters*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Embeddings for word & seq tags
        self.word_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=word_emb_dim)
        self.seq_emb = nn.Embedding(num_embeddings=seq_tag_size, embedding_dim=seq_emb_dim)

        # Figure out CNN output dimension
        dummy_time = 16
        x_dummy = torch.randn(1, 1, dummy_time, num_features)
        with torch.no_grad():
            out_dummy = self.cnn(x_dummy)
        c_out = out_dummy.size(1)   # channels
        t_out = out_dummy.size(2)   # time dimension after pooling
        f_out = out_dummy.size(3)   # freq dimension after pooling
        cnn_out_dim = c_out * f_out

        # LSTM input => [CNN output + word_emb + seq_emb]
        self.lstm_input_dim = cnn_out_dim + word_emb_dim + seq_emb_dim

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        lstm_output_dim = hidden_size * (2 if bidirectional else 1)

        self.fc1 = nn.Linear(lstm_output_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, acoustic_batch, word_batch, seq_batch, lengths):
        """
        acoustic_batch => (B, T, F)
        word_batch     => (B, T)
        seq_batch      => (B, T)
        lengths        => (B,)
        """
        B, T, F = acoustic_batch.size()

        # CNN front-end
        x = acoustic_batch.unsqueeze(1)  # => (B,1,T,F)
        x = self.cnn(x)                  # => (B, channels, T//4, F//4) after 2 pools
        _, c_out, t_out, f_out = x.size()
        x = x.permute(0, 2, 1, 3).reshape(B, t_out, c_out*f_out)

        # Downsample factor => T//t_out (should be 4 if properly padded)
        downsample_factor = T // t_out
        if downsample_factor * t_out != T:
            raise ValueError(f"Time dimension mismatch after pooling: T={T}, t_out={t_out}")

        # Group word/seq in chunks of `downsample_factor`
        grouped_word_list = []
        grouped_seq_list = []
        for b_idx in range(B):
            w_row = word_batch[b_idx]
            s_row = seq_batch[b_idx]
            row_w = []
            row_s = []
            for chunk_start in range(0, T, downsample_factor):
                chunk_end = chunk_start + downsample_factor
                # pick the last item in the chunk
                w_val = w_row[chunk_end - 1]
                s_val = s_row[chunk_end - 1]
                row_w.append(w_val.unsqueeze(0))
                row_s.append(s_val.unsqueeze(0))

            w_out = torch.cat(row_w, dim=0)  # => (t_out,)
            s_out = torch.cat(row_s, dim=0)
            grouped_word_list.append(w_out.unsqueeze(0))
            grouped_seq_list.append(s_out.unsqueeze(0))

        grouped_word_batch = torch.cat(grouped_word_list, dim=0)  # => (B, t_out)
        grouped_seq_batch = torch.cat(grouped_seq_list, dim=0)    # => (B, t_out)

        # Embeddings
        w_embed = self.word_emb(grouped_word_batch)  # (B, t_out, word_emb_dim)
        s_embed = self.seq_emb(grouped_seq_batch)    # (B, t_out, seq_emb_dim)
        ws_cat = torch.cat([w_embed, s_embed], dim=-1)

        # Combine CNN output + embeddings
        combined = torch.cat([x, ws_cat], dim=-1)

        # LSTM (packed sequence)
        lengths_down = torch.clamp(lengths // downsample_factor, min=1, max=t_out)
        packed = nn.utils.rnn.pack_padded_sequence(
            combined, lengths_down.cpu(),
            batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        unpacked, lens_unpacked = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # Grab the last valid output from each sequence
        out_final = []
        for i in range(B):
            seq_len_i = lens_unpacked[i] - 1
            out_final.append(unpacked[i, seq_len_i, :])
        out_final = torch.stack(out_final, dim=0)

        # FC layers
        x = self.fc1(out_final)
        x = Fnn.relu(x)
        x = Fnn.dropout(x, p=self.dropout, training=self.training)
        logits = self.fc2(x)
        return logits


###############################################################################
# 2. Helper functions: loading the checkpoint, and single-CSV inference
###############################################################################
def load_model_checkpoint(model_path, device):
    """
    Loads the saved model weights + architecture hyperparams from `model_path`.
    Must match your training-time checkpoint dictionary structure.
    """
    checkpoint = torch.load(model_path, map_location=device)

    # Unpack saved hyperparameters
    word_to_idx = checkpoint["word_to_idx"]
    seq_to_idx = checkpoint["seq_to_idx"]
    num_features = checkpoint["num_features"]
    cnn_filters = checkpoint["cnn_filters"]
    hidden_size = checkpoint["hidden_size"]
    num_layers = checkpoint["num_layers"]
    num_classes = checkpoint["num_classes"]
    dropout = checkpoint["dropout"]
    bidirectional = checkpoint["bidirectional"]
    vocab_size = checkpoint["vocab_size"]
    seq_tag_size = checkpoint["seq_tag_size"]
    word_emb_dim = checkpoint["word_emb_dim"]
    seq_emb_dim = checkpoint["seq_emb_dim"]

    # print("Loading best model with the following parameters:")
    # print("Word to index:", word_to_idx)
    # print("Sequence to index:", seq_to_idx)
    # print("Number of features:", num_features)
    # print("CNN filters:", cnn_filters)
    # print("Hidden size:", hidden_size)
    # print("Number of layers:", num_layers)
    # print("Number of classes:", num_classes)
    # print("Dropout:", dropout)
    # print("Bidirectional:", bidirectional)
    # print("Vocabulary size:", vocab_size)
    # print("Sequence tag size:", seq_tag_size)
    # print("Word embedding dimension:", word_emb_dim)
    # print("Sequence embedding dimension:", seq_emb_dim)

    model = CNNLSTMClassifier(
        num_features=num_features,
        cnn_filters=cnn_filters,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        vocab_size=vocab_size,
        seq_tag_size=seq_tag_size,
        word_emb_dim=word_emb_dim,
        seq_emb_dim=seq_emb_dim,
        bidirectional=bidirectional
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.lstm.flatten_parameters()

    model.eval()

    # Attach dictionaries for text embeddings
    model.word_to_idx = word_to_idx
    model.seq_to_idx = seq_to_idx

    return model


def pad_to_multiple_of_4(np_array):
    """
    Given we have 3 MaxPool2d(kernel_size=2) layers in the CNN,
    we must ensure T is multiple of 8 for no dimension mismatch.
    """
    T = np_array.shape[0]
    remainder = T % 4
    if remainder == 0:
        return np_array
    needed = 4 - remainder
    pad_shape = (needed,) + np_array.shape[1:]
    pad_frames = np.zeros(pad_shape, dtype=np_array.dtype)
    return np.concatenate([np_array, pad_frames], axis=0)


def infer_single_csv(
    csv_path,
    model,
    device,
    feature_prefixes=("pca_", "mfcc_mean_", "mfcc_var_", "cep_mean_", "cep_var_"),
    word_col="word",
    seq_col="word_seq"
):
    """
    Preprocess a single CSV to match the model's feature expectations,
    pad the time dimension to multiple of 4, then get the classification output.
    """
    df = pd.read_csv(csv_path)
    # Sort by ascending time, if present
    if "time_sec" in df.columns:
        df = df.sort_values(by="time_sec", ascending=True)

    # Replace infinities, fill NaNs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Clip columns as in training
    if "pitch_hz" in df.columns:
        df["pitch_hz"] = df["pitch_hz"].clip(0, 2000)
    if "spec_bw" in df.columns:
        df["spec_bw"] = df["spec_bw"].clip(0, 8000)
    if "jitter" in df.columns:
        df["jitter"] = df["jitter"].clip(0, 1.0)
    if "shimmer" in df.columns:
        df["shimmer"] = df["shimmer"].clip(0, 1.0)
    if "word_dur" in df.columns:
        df["word_dur"] = df["word_dur"].clip(0, 5.0)

    # Additional numeric columns that match your new training code
    additional_cols = [
        "pitch_hz", "energy", "spec_bw", "jitter", "shimmer", "hnr",
        "f1_mean", "f1_std", "f2_mean", "f2_std", "f3_mean", "f3_std",
        "f4_mean", "f4_std", "word_dur"
    ]

    # Gather feature columns by prefix
    feature_cols = [c for c in df.columns if any(c.startswith(pref) for pref in feature_prefixes)]
    # Also add the explicitly named columns (if present)
    for col in additional_cols:
        if col in df.columns:
            feature_cols.append(col)

    # Numeric features
    feats_np = df[feature_cols].values.astype(np.float32)

    # Pad so T is multiple of 8
    feats_np = pad_to_multiple_of_4(feats_np)
    T = feats_np.shape[0]

    # Convert words/seq => IDs
    words = df[word_col].values if word_col in df.columns else ["-"] * len(df)
    seqs = df[seq_col].values if seq_col in df.columns else ["-"] * len(df)

    # After padding feats, pad word/seq arrays if needed
    remainder = T - len(words)
    if remainder > 0:
        words = list(words) + ["-"] * remainder
        seqs = list(seqs) + ["-"] * remainder

    # Convert to IDs using the model dictionaries
    word_ids = [model.word_to_idx.get(w, 0) for w in words]
    seq_ids = [model.seq_to_idx.get(s, 0) for s in seqs]

    # Build Tensors
    feats_tensor = torch.tensor(feats_np, dtype=torch.float32).unsqueeze(0).to(device)  # (1,T,F)
    word_tensor = torch.tensor(word_ids, dtype=torch.long).unsqueeze(0).to(device)      # (1,T)
    seq_tensor = torch.tensor(seq_ids, dtype=torch.long).unsqueeze(0).to(device)        # (1,T)
    lengths_tensor = torch.tensor([T], dtype=torch.long).to(device)                     # (1,)

    model.eval()
    with torch.no_grad():
        logits = model(feats_tensor, word_tensor, seq_tensor, lengths=lengths_tensor)
        probs = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_class].item()

    # Convert predicted class => string label
    label_str = "Parkinsons" if predicted_class == 1 else "Normal"
    return label_str, confidence


###############################################################################
# 3. Main example
###############################################################################
def main():
    """
    Example usage:
      python predict.py
    """
    
    # Decide on your device
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon).")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA (GPU).")
    else:
        print("Using CPU.")

    # Load the trained model checkpoint
    model_path = "best_model_cnn_lstm.pth"  # Adjust path as needed
    model = load_model_checkpoint(model_path, device)

    # Example: run inference on one Normal CSV
    csv_path = "test/normal/video_2010_clip1.csv"
    label, conf = infer_single_csv(
        csv_path=csv_path,
        model=model,
        device=device,
        feature_prefixes=("pca_", "mfcc_mean_", "mfcc_var_", "cep_mean_", "cep_var_"),
        word_col="word",
        seq_col="word_seq"
    )

    print("----------------------------------------------------")
    print(f"File: {csv_path}")
    print(f"Predicted: {label} (Confidence: {conf*100:.2f}%)")
    print("----------------------------------------------------")

    # Example: run inference on one Parkinsons CSV
    csv_path = "test/parkinsons/video_2003_clip4.csv"
    label, conf = infer_single_csv(
        csv_path=csv_path,
        model=model,
        device=device,
        feature_prefixes=("pca_", "mfcc_mean_", "mfcc_var_", "cep_mean_", "cep_var_"),
        word_col="word",
        seq_col="word_seq"
    )

    print("----------------------------------------------------")
    print(f"File: {csv_path}")
    print(f"Predicted: {label} (Confidence: {conf*100:.2f}%)")
    print("----------------------------------------------------")


if __name__ == "__main__":
    main()
