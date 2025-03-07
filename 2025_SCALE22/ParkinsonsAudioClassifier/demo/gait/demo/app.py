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
                analyze_file = "test/normal.csv"
            elif numbers[0] == "2":
                analyze_file = "test/parkinsons.csv"
            else:
                print('Unknown patient number.')
                return

            # get the results
            global model
            global global_scaler
            diagnoses_result = predict_on_new_csv(model, global_scaler, analyze_file)

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
    model_name = "parkinsons_classifier_model_complete.pth"
    model_path = Path(model_name)

    # Load our model
    checkpoint = torch.load(model_name)

    # Extract model parameters
    input_size = checkpoint["input_size"]
    hidden_size = checkpoint["hidden_size"]
    num_layers = checkpoint["num_layers"]
    num_classes = checkpoint["num_classes"]

    global global_scaler
    global_scaler = checkpoint["global_scaler"]  # Load the scaler

    # Reconstruct the model
    global model
    model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
    # Load the saved state dictionary
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

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
