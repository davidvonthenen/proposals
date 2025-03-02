import os
import glob
import random
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as Fnn  # Avoid overshadowing 'F'
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

# --------------------
#  Device Configuration
# --------------------
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
# 1. Data Gathering & Helpers
###############################################################################
def gather_csv_files(data_root):
    """
    gather_csv_files(data_root)
    ---------------------------
    Given a directory structure like:
        data_root/Parkinsons/*.csv
        data_root/Normal/*.csv

    Returns a list of (csv_path, label):
      label=1 => "Parkinsons"
      label=0 => "Normal"

    Parameters
    ----------
    data_root : str
        The parent folder containing subfolders for each class.

    Returns
    -------
    csv_label_pairs : list of (str, int)
        Each element is (path_to_csv, label).
    """
    csv_label_pairs = []

    # Parkinsons
    parkinsons_dir = os.path.join(data_root, "Parkinsons")
    parkinsons_files = glob.glob(os.path.join(parkinsons_dir, "*.csv"))
    for fpath in parkinsons_files:
        csv_label_pairs.append((fpath, 1))  # label=1 => Parkinsons

    # Normal
    normal_dir = os.path.join(data_root, "Normal")
    normal_files = glob.glob(os.path.join(normal_dir, "*.csv"))
    for fpath in normal_files:
        csv_label_pairs.append((fpath, 0))  # label=0 => Normal

    return csv_label_pairs


###############################################################################
# 2. Dataset Class with Global Feature Scaling
###############################################################################
class ParkinsonsNormalCsvDataset(Dataset):
    """
    ParkinsonsNormalCsvDataset
    --------------------------
    Custom Torch Dataset that:
      - Loads each CSV, which represents one utterance with multiple frames.
      - Extracts acoustic/prosodic features (like PCA columns, MFCC stats, etc.).
      - Applies global StandardScaler across all CSVs to standardize features.
      - Encodes words/word_seq columns into IDs if present.

    Implementation details:
      - We do two passes over the CSV paths:
        1) `_collect_for_scaler()`: Gather all numeric data to fit StandardScaler.
        2) `_prepare_data()`: Actually load + scale + store each utterance.
      - Also handles zero-padding the time dimension to be divisible by 4 so
        the CNN's pooling doesn't run out of frames.
    """
    def __init__(
        self,
        csv_label_pairs,
        feature_prefixes=("pca_", "mfcc_mean_", "mfcc_var_", "cep_mean_", "cep_var_"),
        word_col="word",
        word_seq_col="word_seq",
        word_to_idx=None,
        seq_to_idx=None,
        do_augmentation=False
    ):
        """
        Parameters
        ----------
        csv_label_pairs : list of (str, int)
            Each element is (path_to_csv, label).
        feature_prefixes : tuple of str
            Column prefixes used to identify numeric feature columns in each CSV.
        word_col : str
            Name of the column that stores the recognized word token.
        word_seq_col : str
            Name of the column that indicates whether it's START/CONTINUE/'-'.
        word_to_idx : dict, optional
            Mapping of words -> integer IDs (for embeddings). If None, built here.
        seq_to_idx : dict, optional
            Mapping of sequence tags -> integer IDs. If None, built here.
        do_augmentation : bool
            If True, performs simple feature-level augmentation by random scaling.
        """
        super().__init__()
        self.csv_label_pairs = csv_label_pairs
        self.feature_prefixes = feature_prefixes
        self.word_col = word_col
        self.word_seq_col = word_seq_col
        self.do_augmentation = do_augmentation

        # If user doesn't supply a dictionary for words, we create it
        self.word_to_idx = word_to_idx if word_to_idx else {}
        self.seq_to_idx = seq_to_idx if seq_to_idx else {}

        # Will hold (feats, word_ids, seq_ids, label) for each CSV
        self.data = []

        # 1) Gather numeric data across all CSVs for scaling
        all_feature_rows = []
        self._collect_for_scaler(all_feature_rows)

        # 2) Fit the StandardScaler (global for the entire dataset)
        self.scaler = None
        if len(all_feature_rows) > 0:
            self.scaler = StandardScaler()
            self.scaler.fit(np.vstack(all_feature_rows))

        # 3) Load the data fully (scaled + padded)
        self._prepare_data()


    def _collect_for_scaler(self, all_feature_rows):
        """
        _collect_for_scaler(all_feature_rows)
        -------------------------------------
        First pass: read each CSV, accumulate its numeric rows into 'all_feature_rows'.
        Also build or update the dictionaries for recognized words/seq tags.

        Parameters
        ----------
        all_feature_rows : list
            We append each utterance's 2D array of numeric features to this list.
        """
        word_set = set()
        seq_set = set()

        for csv_path, _ in self.csv_label_pairs:
            df = pd.read_csv(csv_path)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(0, inplace=True)

            # Collect any new words or seq tags
            if self.word_col in df.columns:
                word_set.update(df[self.word_col].unique())
            if self.word_seq_col in df.columns:
                seq_set.update(df[self.word_seq_col].unique())

            # Clamp suspiciously large or small values
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

            # Additional numeric columns to consider
            additional_cols = [
                "pitch_hz", "energy", "spec_bw", "jitter", "shimmer",
                "hnr",
                "f1_mean", "f1_std", "f2_mean", "f2_std",
                "f3_mean", "f3_std", "f4_mean", "f4_std",
                "word_dur",
            ]

            # Identify the columns matching our feature prefixes
            feat_cols = [
                c for c in df.columns
                if any(c.startswith(pref) for pref in self.feature_prefixes)
            ]

            # Add the explicitly named columns if they exist
            for col in additional_cols:
                if col in df.columns:
                    feat_cols.append(col)

            # Build a 2D array of just the numeric features for scaling
            feats_np = df[feat_cols].values.astype(np.float32)
            all_feature_rows.append(feats_np)

        # Build dictionary for words if not already provided
        if not self.word_to_idx and word_set:
            self.word_to_idx = self._build_word_dict(word_set)

        # Build dictionary for seq tags if not already provided
        if not self.seq_to_idx and seq_set:
            self.seq_to_idx = self._build_seq_dict(seq_set)


    def _prepare_data(self):
        """
        _prepare_data()
        ---------------
        Second pass: Load each CSV for real, scale the numeric features,
        encode words to IDs, pad the time dimension, store final data
        into self.data for retrieval by __getitem__().
        """
        for csv_path, label in self.csv_label_pairs:
            df = pd.read_csv(csv_path)

            # print(df[["pitch", "energy", "spectral_bandwidth", "formant_f1", "formant_f2", "formant_f3", "jitter", "shimmer", "hnr", "word_duration"]].describe())
            # print("Any NaNs? ", df.isna().any())
            # print("Any Infs? ", np.isinf(df).any().any())

            # Sort by time_sec in ascending order to preserve temporal structure
            if "time_sec" in df.columns:
                df = df.sort_values(by="time_sec", ascending=True)

            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(0, inplace=True)

            # Same clamping logic
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

            # Additional aggregator columns that might be present
            additional_cols = [
                "pitch_hz", "energy", "spec_bw", "jitter", "shimmer",
                "hnr",
                "f1_mean", "f1_std", "f2_mean", "f2_std",
                "f3_mean", "f3_std", "f4_mean", "f4_std",
                "word_dur",
            ]
            feat_cols = [
                c for c in df.columns
                if any(c.startswith(pref) for pref in self.feature_prefixes)
            ]
	        # print("-----------> Feature columns:", feature_cols)
            for col in additional_cols:
                if col in df.columns:
                    feat_cols.append(col)
            # print("-----------> Feature columns:", feature_cols)

            # Convert to numeric array
            feats_np = df[feat_cols].values.astype(np.float32)

            # Scale using the previously fitted StandardScaler
            if self.scaler is not None:
                shape_orig = feats_np.shape
                feats_np = feats_np.reshape(-1, shape_orig[-1])
                feats_np = self.scaler.transform(feats_np)
                feats_np = feats_np.reshape(shape_orig)

            # Optional data augmentation
            if self.do_augmentation and random.random() < 0.3:
                # Multiply by (1 + Gaussian noise) ~ 10% variation
                feats_np = feats_np * (1.0 + 0.1 * np.random.randn(*feats_np.shape))

            # Convert word -> ID
            if self.word_col in df.columns:
                word_ids = [self.word_to_idx.get(w, 0) for w in df[self.word_col].values]
            else:
                word_ids = [0] * len(df)

            # Convert word_seq -> ID
            if self.word_seq_col in df.columns:
                seq_ids = [self.seq_to_idx.get(s, 0) for s in df[self.word_seq_col].values]
            else:
                seq_ids = [0] * len(df)

            # Pad to multiple of 4 in time dimension
            feats_np, word_ids_np, seq_ids_np = self._pad_to_multiple_of_4(
                feats_np,
                np.array(word_ids, dtype=np.int64),
                np.array(seq_ids, dtype=np.int64)
            )

            self.data.append((feats_np, word_ids_np, seq_ids_np, label))


    def _pad_to_multiple_of_4(self, feats, word_ids, seq_ids):
        """
        _pad_to_multiple_of_4(feats, word_ids, seq_ids)
        -----------------------------------------------
        Ensures the time dimension T is a multiple of 4 by zero-padding
        (important for a CNN with 2 levels of pooling).

        Returns
        -------
        feats, word_ids, seq_ids : np.array
            Padded arrays in time dimension if needed.
        """
        T = feats.shape[0]
        remainder = T % 4
        if remainder == 0:
            return feats, word_ids, seq_ids

        needed = 4 - remainder

        # Pad feats (shape: (T, D))
        pad_feats = np.zeros((needed, feats.shape[1]), dtype=feats.dtype)
        feats = np.concatenate([feats, pad_feats], axis=0)

        # Pad word_ids (shape: (T,))
        pad_word_ids = np.zeros((needed,), dtype=word_ids.dtype)
        word_ids = np.concatenate([word_ids, pad_word_ids], axis=0)

        # Pad seq_ids (shape: (T,))
        pad_seq_ids = np.zeros((needed,), dtype=seq_ids.dtype)
        seq_ids = np.concatenate([seq_ids, pad_seq_ids], axis=0)

        return feats, word_ids, seq_ids

    def _build_word_dict(self, word_set):
        """
        _build_word_dict(word_set)
        --------------------------
        Builds a dictionary from words to integer IDs.
        0 => unknown/'-'

        Returns
        -------
        dict
            word -> integer ID
        """
        # Remove '-' from the set so it can be assigned 0
        cleaned = sorted(word_set - {"-"})
        w2i = {"-": 0}
        i = 1
        for w in cleaned:
            w2i[w] = i
            i += 1
        return w2i

    def _build_seq_dict(self, seq_set):
        """
        _build_seq_dict(seq_set)
        ------------------------
        Build dictionary for word_seq: e.g. '-', 'START', 'CONTINUE'
        0 => unknown or '-'
        """
        cleaned = sorted(seq_set)
        s2i = {}
        i = 0
        for s in cleaned:
            s2i[s] = i
            i += 1
        return s2i

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves the (feats_tensor, word_ids_tensor, seq_ids_tensor, label_tensor)
        for a given utterance index.
        """
        feats_np, word_ids_np, seq_ids_np, label = self.data[idx]
        feats_tensor = torch.tensor(feats_np, dtype=torch.float32)
        word_ids_tensor = torch.tensor(word_ids_np, dtype=torch.long)
        seq_ids_tensor = torch.tensor(seq_ids_np, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return feats_tensor, word_ids_tensor, seq_ids_tensor, label_tensor


###############################################################################
# 3. Collate Function
###############################################################################
def utterance_collate_fn(batch):
    """
    utterance_collate_fn(batch)
    ---------------------------
    Collate function to combine a list of samples into a batch for the DataLoader.

    Each item in batch = (feats_tensor, word_ids, seq_ids, label).
    We zero-pad to the maximum sequence length in the batch if needed, but in this
    pipeline, each utterance has already been padded to a multiple of 4. So we just
    align them to the largest T in the batch.

    Returns
    -------
    acoustic_batch : (B, T_max, F) FloatTensor
    word_batch : (B, T_max) LongTensor
    seq_batch : (B, T_max) LongTensor
    labels : (B,) LongTensor
    lengths : (B,) LongTensor
        The original length of each utterance (before padding).
    """
    lengths = [b[0].size(0) for b in batch]
    max_len = max(lengths)
    B = len(batch)
    acoustic_dim = batch[0][0].size(1)

    acoustic_batch = torch.zeros((B, max_len, acoustic_dim), dtype=torch.float32)
    word_batch = torch.zeros((B, max_len), dtype=torch.long)
    seq_batch = torch.zeros((B, max_len), dtype=torch.long)
    labels = torch.zeros((B,), dtype=torch.long)
    lengths_t = torch.tensor(lengths, dtype=torch.long)

    for i, (feats, w_ids, s_ids, lab) in enumerate(batch):
        seq_len = feats.size(0)
        acoustic_batch[i, :seq_len, :] = feats
        word_batch[i, :seq_len] = w_ids
        seq_batch[i, :seq_len] = s_ids
        labels[i] = lab

    return acoustic_batch, word_batch, seq_batch, labels, lengths_t


###############################################################################
# 4. CNN+LSTM Classifier
###############################################################################
class CNNLSTMClassifier(nn.Module):
    """
    CNNLSTMClassifier
    -----------------
    A hybrid architecture with:
      - A CNN front-end operating in (time x feature) space for each utterance
        (with 2D pooling to downsample time dimension).
      - An LSTM back-end that takes the CNN outputs plus embeddings for words/seq tags.
      - Outputs a binary classification (Parkinsons vs Normal).

    The forward pass handles:
      1) Convolution + pooling layers => downsample time dimension by factor of ~4
      2) Summarize the "word" and "seq" tokens over the same downsample factor
      3) LSTM => final hidden => fully connected layer => logits
    """
    def __init__(
        self,
        num_features=64,
        cnn_filters=32,
        hidden_size=128,
        num_layers=2,
        num_classes=2,
        dropout=0.5,
        vocab_size=100,        # distinct "word" tokens
        seq_tag_size=3,        # e.g. '-', 'START', 'CONTINUE'
        word_emb_dim=16,
        seq_emb_dim=4,
        use_pack_sequence=True,
        bidirectional=True
    ):
        super().__init__()
        self.use_pack_sequence = use_pack_sequence
        self.num_features = num_features
        self.cnn_filters = cnn_filters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.bidirectional = bidirectional

        # 2 successive MaxPool2d => /4 in time dimension
        # Input shape for CNN is (B, 1, T, F)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(cnn_filters, cnn_filters * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Embeddings for 'word' and 'seq' tokens
        self.word_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=word_emb_dim)
        self.seq_emb = nn.Embedding(num_embeddings=seq_tag_size, embedding_dim=seq_emb_dim)

        # Determine dimension after CNN
        # We run a dummy input through the CNN just to figure out the shape
        dummy_time = 16
        x_dummy = torch.randn(1, 1, dummy_time, num_features)
        with torch.no_grad():
            out_dummy = self.cnn(x_dummy)
        c_out = out_dummy.size(1)
        t_out = out_dummy.size(2)
        f_out = out_dummy.size(3)
        cnn_out_dim = c_out * f_out  # We flatten the last 2D dimension into 1D

        # LSTM input dimension = CNN features + word_emb + seq_emb
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

        # Final classification layers
        self.fc1 = nn.Linear(lstm_output_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, acoustic_batch, word_batch, seq_batch, lengths=None):
        """
        forward(...)
        -----------
        acoustic_batch : (B, T, F)
            B = batch size, T = time frames, F = feature dimension
        word_batch : (B, T)
            Word token IDs
        seq_batch : (B, T)
            Sequence tag IDs
        lengths : (B,)
            Length of each utterance before padding (optional).

        Returns
        -------
        logits : (B, num_classes)
            Classification scores for each utterance.
        """
        B, T, F = acoustic_batch.size()

        # 1) CNN front-end
        x = acoustic_batch.unsqueeze(1)  # => (B, 1, T, F)
        x = self.cnn(x)                  # => (B, Channels, T//4, F//4) if 2 levels of pooling
        _, c_out, t_out, f_out = x.size()
        # Rearrange => (B, t_out, c_out*f_out)
        x = x.permute(0, 2, 1, 3).reshape(B, t_out, c_out * f_out)
        downsample_factor = T // t_out  # Typically T_out = T/4

        # 2) Group word/seq tokens over the same downsample factor
        grouped_word_list = []
        grouped_seq_list = []
        for b_idx in range(B):
            w_row = word_batch[b_idx]
            s_row = seq_batch[b_idx]
            row_w_chunks = []
            row_s_chunks = []
            for chunk_start in range(0, T, downsample_factor):
                chunk_end = chunk_start + downsample_factor
                chunk_w = w_row[chunk_start:chunk_end]
                chunk_s = s_row[chunk_start:chunk_end]
                # We'll pick the "last" token in that chunk
                w_val = chunk_w[-1]
                s_val = chunk_s[-1]
                row_w_chunks.append(w_val.unsqueeze(0))
                row_s_chunks.append(s_val.unsqueeze(0))

            w_out = torch.cat(row_w_chunks, dim=0)  # => shape(t_out,)
            s_out = torch.cat(row_s_chunks, dim=0)  # => shape(t_out,)
            grouped_word_list.append(w_out.unsqueeze(0))  # => (1, t_out)
            grouped_seq_list.append(s_out.unsqueeze(0))   # => (1, t_out)

        grouped_word_batch = torch.cat(grouped_word_list, dim=0)  # => (B, t_out)
        grouped_seq_batch = torch.cat(grouped_seq_list, dim=0)    # => (B, t_out)

        # 3) Embeddings
        w_embed = self.word_emb(grouped_word_batch)  # => (B, t_out, word_emb_dim)
        s_embed = self.seq_emb(grouped_seq_batch)    # => (B, t_out, seq_emb_dim)
        ws_cat = torch.cat([w_embed, s_embed], dim=-1)  # => (B, t_out, word_emb_dim + seq_emb_dim)

        # 4) Concatenate CNN features + embeddings => LSTM input
        combined = torch.cat([x, ws_cat], dim=-1)  # => (B, t_out, self.lstm_input_dim)

        # If using pack_sequence, we pack the sequences for more efficient LSTM
        if self.use_pack_sequence and lengths is not None:
            lengths_down = torch.clamp(lengths // downsample_factor, min=1, max=t_out)
            packed = nn.utils.rnn.pack_padded_sequence(
                combined, lengths_down.cpu(),
                batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            unpacked, lens_unpacked = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

            # Grab the last valid output for each sequence
            out_final = []
            for i in range(B):
                seq_len_i = lens_unpacked[i] - 1
                out_final.append(unpacked[i, seq_len_i, :])
            out_final = torch.stack(out_final, dim=0)
        else:
            lstm_out, _ = self.lstm(combined)
            out_final = lstm_out[:, -1, :]

        # 5) Final classification
        x = self.fc1(out_final)         # => (B, hidden_size)
        x = Fnn.relu(x)
        x = Fnn.dropout(x, p=self.dropout, training=self.training)
        logits = self.fc2(x)           # => (B, num_classes)
        return logits


###############################################################################
# 5. Training & Evaluation
###############################################################################
def train_model(model, train_loader, val_loader, num_epochs=30, learning_rate=0.001, patience=5):
    """
    train_model(...)
    ---------------
    Trains the given model using the provided DataLoaders for train and validation.
    Uses an Adam optimizer with weight decay and learning rate scheduling.
    Implements early stopping based on validation loss.

    Parameters
    ----------
    model : nn.Module
        The neural network (CNN+LSTM) to train.
    train_loader : DataLoader
        DataLoader for the training set.
    val_loader : DataLoader
        DataLoader for the validation set.
    num_epochs : int
        Maximum number of epochs to train.
    learning_rate : float
        Initial learning rate for the optimizer.
    patience : int
        Number of epochs to wait without improvement in val_loss before stopping.

    Returns
    -------
    train_losses : list of float
        Per-epoch average training loss.
    val_losses : list of float
        Per-epoch average validation loss.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    no_improve_count = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0
        for acoustic_batch, word_batch, seq_batch, y_batch, lengths in train_loader:
            acoustic_batch = acoustic_batch.to(device)
            word_batch = word_batch.to(device)
            seq_batch = seq_batch.to(device)
            y_batch = y_batch.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            outputs = model(acoustic_batch, word_batch, seq_batch, lengths=lengths)
            loss = criterion(outputs, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for acoustic_batch, word_batch, seq_batch, y_batch, lengths in val_loader:
                acoustic_batch = acoustic_batch.to(device)
                word_batch = word_batch.to(device)
                seq_batch = seq_batch.to(device)
                y_batch = y_batch.to(device)
                lengths = lengths.to(device)

                outputs = model(acoustic_batch, word_batch, seq_batch, lengths=lengths)
                loss = criterion(outputs, y_batch)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Step the scheduler
        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            # Save the best model state
            torch.save(model.state_dict(), "best_model_cnn_lstm.pth")

            # print("Saving best model...")
            # print("Word to index:", model.word_to_idx)
            # print("Sequence to index:", model.seq_to_idx)
            # print("Number of features:", model.num_features)
            # print("CNN filters 1:", model.cnn[0].out_channels)
            # print("CNN filters 2:", model.cnn_filters)
            # print("Hidden size:", model.hidden_size)
            # print("Number of layers:", model.num_layers)
            # print("Number of classes:", model.num_classes)
            # print("Dropout:", model.dropout)
            # print("Bidirectional:", model.bidirectional)
            # print("Vocabulary size:", model.word_emb.num_embeddings)
            # print("Sequence tag size:", model.seq_emb.num_embeddings)
            # print("Word embedding dimension:", model.word_emb.embedding_dim)
            # print("Sequence embedding dimension:", model.seq_emb.embedding_dim)

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "word_to_idx": model.word_to_idx,
                "seq_to_idx": model.seq_to_idx,
                "num_features": model.num_features,
                "cnn_filters": model.cnn_filters,
                "hidden_size": model.hidden_size,
                "num_layers": model.num_layers,
                "num_classes": model.num_classes,
                "dropout": model.dropout,
                "bidirectional": model.bidirectional,
                "vocab_size": model.word_emb.num_embeddings,
                "seq_tag_size": model.seq_emb.num_embeddings,
                "word_emb_dim": model.word_emb.embedding_dim,
                "seq_emb_dim": model.seq_emb.embedding_dim
            }

            torch.save(checkpoint, "best_model_cnn_lstm.pth")

            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print("Early stopping triggered.")
                break

    # Load the best model
    checkpoint = torch.load("best_model_cnn_lstm.pth")

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
    # print("Sequence embedding dimension:",seq_emb_dim)

    model.load_state_dict(checkpoint["model_state_dict"])

    return train_losses, val_losses


def test_model(model, test_loader):
    """
    test_model(model, test_loader)
    -----------------------------
    Evaluates the model on a test set.

    Parameters
    ----------
    model : nn.Module
        Trained CNN+LSTM classifier.
    test_loader : DataLoader
        DataLoader for the test set.

    Returns
    -------
    accuracy : float
        Classification accuracy as a fraction (0..1).
    """
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for acoustic_batch, word_batch, seq_batch, y_batch, lengths in test_loader:
            acoustic_batch = acoustic_batch.to(device)
            word_batch = word_batch.to(device)
            seq_batch = seq_batch.to(device)
            y_batch = y_batch.to(device)
            lengths = lengths.to(device)

            outputs = model(acoustic_batch, word_batch, seq_batch, lengths=lengths)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(y_batch.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return accuracy


###############################################################################
# 6. Main
###############################################################################
def main():
    """
    main()
    ------
    Example usage pipeline:
      1. Gathers CSV files from data_root/Parkinsons and data_root/Normal.
      2. Loads them into a ParkinsonsNormalCsvDataset (scales globally).
      3. Splits dataset into train, val, test.
      4. Trains a CNN+LSTM classifier with early stopping.
      5. Tests the model and prints accuracy.
      6. Plots training/validation losses.
    """
    data_root = "processed"  # Directory containing "Parkinsons" and "Normal" subfolders
    csv_label_pairs = gather_csv_files(data_root)

    # Create dataset (scaling is done internally)
    dataset = ParkinsonsNormalCsvDataset(
        csv_label_pairs=csv_label_pairs,
        feature_prefixes=("pca_", "mfcc_mean_", "mfcc_var_", "cep_mean_", "cep_var_"),
        word_col="word",
        word_seq_col="word_seq",
        do_augmentation=False
    )

    # Shuffle and split
    num_samples = len(dataset)
    all_indices = list(range(num_samples))
    random.shuffle(all_indices)

    # 80% train, 10% val, 10% test
    split1 = int(0.8 * num_samples)
    split2 = int(0.9 * num_samples)
    train_idx = all_indices[:split1]
    val_idx = all_indices[split1:split2]
    test_idx = all_indices[split2:]

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    test_subset = Subset(dataset, test_idx)

    batch_size = 8

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=utterance_collate_fn
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=utterance_collate_fn
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=utterance_collate_fn
    )

    # Derive #acoustic features from the first sample in dataset
    feats_ex, word_ex, seq_ex, label_ex = dataset[0]
    num_acoustic_features = feats_ex.size(1)

    # Build the classifier
    vocab_size = len(dataset.word_to_idx)
    seq_tag_size = len(dataset.seq_to_idx)

    model = CNNLSTMClassifier(
        num_features=num_acoustic_features,
        vocab_size=vocab_size,
        seq_tag_size=seq_tag_size,
    ).to(device)

    # Store dictionaries on model for reference (useful when saving)
    model.word_to_idx = dataset.word_to_idx
    model.seq_to_idx = dataset.seq_to_idx

    # Train
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # Test
    test_accuracy = test_model(model, test_loader)

    # Plot training & validation loss curves
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
