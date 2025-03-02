import json
import multiprocessing
import time
import platform
import os
import random
import warnings
from pathlib import Path
import re

import torch
import pandas as pd
import numpy as np
from torch import nn
import torch.nn.functional as Fnn

from deepgram.utils import verboselogs
from websockets.sync.server import serve
from flask import Flask, send_from_directory

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    SpeakWSOptions,
    SpeakWebSocketEvents,
    LiveTranscriptionEvents,
    LiveOptions,
)

import openai

# Disable future warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

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
BOLD = "\033[1m"
RESET = "\033[0m"
MAGENTA = "\033[35m"



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

        print(
            MAGENTA
            + f"Prediction for {csv_path}: {label_str}, Probability: {confidence:.4f}\n"
            + RESET
        )
        if label_str == "normal":
            return f"Based on audio analysis, the patent demonstrates normal vocal characteristics with a probability of {confidence * 100:.2f}%"
        return f"Based on audio analysis, the patent demonstrates vocal characteristic of a patient with {label_str} with a probability of {confidence * 100:.2f}%"



app = Flask(__name__, static_folder="./public", static_url_path="/public")

# Maintain global timing for sending WAV headers in TTS
last_time = time.time() - 5

def hello(websocket):
    global last_time
    last_time = time.time() - 5

    # Deepgram
    config = DeepgramClientOptions(
        # verbose=verboselogs.DEBUG,
        options={"keepalive": "true"},
    )
    deepgram: DeepgramClient = DeepgramClient("", config)

    dg_tts_connection = None
    dg_stt_connection = None

    # OpenAI
    openai_client = openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    openai_messages = [
        {
            "role": "system",
            "content": "Your name is Computer. You are a compassionate medical assistant who is here to answer questions in the most concise way possible.\nAvoid using the term diagnoses or a result that's very final. Always limit your responses to 2 sentences or 75 words which ever is less. Never say your own name.",
        }
    ]

    ### TTS Event Handlers ###
    def on_tts_open(self, open, **kwargs):
        print("TTS: Connection Open")

    def on_tts_flush(self, flushed, **kwargs):
        print("\nTTS: Flush\n")
        # After flushing TTS data, send "Flushed" message to client
        flush_message = json.dumps({"type": "Flushed"})
        websocket.send(flush_message)

    def on_tts_binary_data(self, data, **kwargs):
        # Send WAV header only once before the first audio chunk
        global last_time
        if time.time() - last_time > 3:
            # Send WAV header before first chunk
            header = bytes(
                [
                    0x52, 0x49, 0x46, 0x46,  # "RIFF"
                    0x00, 0x00, 0x00, 0x00,
                    0x57, 0x41, 0x56, 0x45,  # "WAVE"
                    0x66, 0x6D, 0x74, 0x20,  # "fmt "
                    0x10, 0x00, 0x00, 0x00,
                    0x01, 0x00,
                    0x01, 0x00,
                    0x80, 0xBB, 0x00, 0x00,
                    0x00, 0xEE, 0x02, 0x00,
                    0x02, 0x00,
                    0x10, 0x00,
                    0x64, 0x61, 0x74, 0x61,  # "data"
                    0x00, 0x00, 0x00, 0x00,
                ]
            )
            websocket.send(header)
            last_time = time.time()

        # Send binary TTS audio to client
        # print("TTS: Sending audio data")
        websocket.send(data)

    def on_tts_close(self, close, **kwargs):
        print("TTS: Connection Closed")

    ### STT Event Handlers ###
    def on_stt_open(self, open, **kwargs):
        print("STT: Connection Open")

    def on_stt_message(self, result, **kwargs):
        sentence = result.channel.alternatives[0].transcript
        if len(sentence) < 8:
            return
        if result.is_final is False:
            return
        
        # key word trigger
        if "Computer" not in sentence:
            print("No Computer")
            return

        # sentence
        print(f"\nSpeech-to-Text: {sentence}\n")

        is_diagnosis = False
        count = 1
        diagnoses_result = "Undetermined at this time."
        if "patient" in sentence:
            # eval
            is_diagnosis = True

            # extract patient id
            numbers = re.findall(r"\d+", sentence)
            if len(numbers) == 0:
                return

            # simulate loading a database
            if numbers[0] == "1":
                analyze_file = "test/normal/video_2010_clip1.csv"
            elif numbers[0] == "2":
                analyze_file = "test/parkinsons/video_2003_clip4.csv"
            else:
                print("Unknown patient number.")
                return

            # get the results
            global model
            global global_scaler
            diagnoses_result = infer_single_csv(analyze_file, model, device)

        # LLM
        # append to the openai messages
        if is_diagnosis:
            openai_messages.append(
                {
                    "role": "user",
                    "content": f"You just obtain the video results from the lab. Break the following results of the analysis to the patient:\n{diagnoses_result}",
                }
            )
        else:
            openai_messages.append({"role": "user", "content": f"{sentence}"})

        # send to ChatGPT
        save_response = ""
        try:
            for response in openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=openai_messages,
                stream=True,
            ):
                # here is the streaming response
                for chunk in response:
                    if chunk[0] == "choices":
                        llm_output = chunk[1][0].delta.content

                        # skip any empty responses
                        if llm_output is None or llm_output == "":
                            continue

                        # save response and append to buffer
                        save_response += llm_output

                        # send to Deepgram TTS
                        dg_tts_connection.send_text(llm_output)

            # print the response
            print(f"\nText-to-Speech: {save_response}\n")

            # append the response to the openai messages
            openai_messages.append(
                {"role": "assistant", "content": f"{save_response}"}
            )
            dg_tts_connection.flush()

        except Exception as e:
            print("LLM Exception:", e)

    def on_stt_close(self, close, **kwargs):
        print("STT: Connection Closed")

    # Start STT connection with encoding=opus to match browser MediaRecorder output
    dg_stt_connection = deepgram.listen.websocket.v("1")

    dg_stt_connection.on(LiveTranscriptionEvents.Open, on_stt_open)
    dg_stt_connection.on(LiveTranscriptionEvents.Transcript, on_stt_message)
    dg_stt_connection.on(LiveTranscriptionEvents.Close, on_stt_close)

    stt_options = LiveOptions(
        model="nova-2",
        language="en-US",
        smart_format=True,
        numerals=True,
    )

    # Create a Deepgram TTS connection based on the provided example
    dg_tts_connection = deepgram.speak.websocket.v("1")

    dg_tts_connection.on(SpeakWebSocketEvents.Open, on_tts_open)
    dg_tts_connection.on(SpeakWebSocketEvents.AudioData, on_tts_binary_data)
    dg_tts_connection.on(SpeakWebSocketEvents.Flushed, on_tts_flush)
    dg_tts_connection.on(SpeakWebSocketEvents.Close, on_tts_close)

    tts_options: SpeakWSOptions = SpeakWSOptions(
        model="aura-asteria-en",
        encoding="linear16",
        sample_rate=48000,
    )

    # Main Loop
    while True:
        try:
            message = websocket.recv()
        except:
            # Client disconnected
            break

        if message is None:
            # sleep for a bit to avoid busy loop
            if app.debug:
                app.logger.debug(
                    "No bytes received from client, sleeping for 0.1 seconds"
                )
            time.sleep(0.1)
            continue

        # if bytes are received, send them to Deepgram STT
        if isinstance(message, bytes):
            # Incoming binary is audio data from client for STT
            # print(f"----> BINARY: {len(message)} bytes")
            dg_stt_connection.send(message)
        else:
            # Incoming text is a command from client
            print(f"----> TEXT: {message}")

            # check for {"type":"transcription_control","action":"stop"}
            if message == '{"type":"transcription_control","action":"stop"}':
                dg_stt_connection.finish()
                dg_tts_connection.finish()
            elif message == '{"type":"transcription_control","action":"start"}':
                # STT
                if dg_stt_connection.start(stt_options) is False:
                    if app.debug:
                        app.logger.debug(
                            "Unable to start Deepgram TTS WebSocket connection"
                        )
                    raise Exception("Unable to start Deepgram STT WebSocket connection")
                else:
                    if app.debug:
                        app.logger.debug("Deepgram STT WebSocket connection started")

                # TTS
                if dg_tts_connection.start(tts_options) is False:
                    if app.debug:
                        app.logger.debug(
                            "Unable to start Deepgram TTS WebSocket connection"
                        )
                    raise Exception("Unable to start Deepgram TTS WebSocket connection")
                else:
                    if app.debug:
                        app.logger.debug("Deepgram TTS WebSocket connection started")

@app.route("/<path:filename>")
def serve_others(filename):
    return send_from_directory(app.static_folder, filename)

@app.route("/assets/<path:filename>")
def serve_image(filename):
    return send_from_directory(app.static_folder, "assets/" + filename)

@app.route("/", methods=["GET"])
def serve_index():
    return app.send_static_file("index.html")

def run_ui():
    app.run(debug=True, use_reloader=False)

def run_ws():
    # Model file name
    model_name = "best_model_cnn_lstm.pth"
    model_path = Path(model_name)

    # Load the trained model checkpoint
    global model
    model = load_model_checkpoint(model_path, device)

    print("Model loaded successfully.")

    with serve(hello, "localhost", 3000) as server:
        server.serve_forever()

if __name__ == "__main__":
    p_flask = multiprocessing.Process(target=run_ui)
    p_ws = multiprocessing.Process(target=run_ws)

    p_flask.start()
    p_ws.start()

    p_flask.join()
    p_ws.join()
