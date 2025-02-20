import time
import os
import threading
import queue
import warnings

import sounddevice as sd

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from kokoro_onnx import Kokoro

from deepgram import (
    Microphone,
)

import openai


#################
# Speech-to-Text
#################

# Microphone instance.
global mic
mic = None

# what device are we using?
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Optionally, to suppress deprecation warnings, uncomment:
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

# Thread-safe queue for audio chunks.
audio_queue = queue.Queue()
stop_event = threading.Event()

# Energy threshold: chunks with RMS energy below this are considered silence.
energy_threshold = 0.010

# Silence duration (in seconds) required to mark the end of a thought.
silence_duration = 1.5

# Determine device: use GPU if available, else CPU.
device = 0 if torch.cuda.is_available() else -1

# Define the model name.
model_name = "openai/whisper-large-v3-turbo"


# Load the Whisper model and processor locally.
# model = WhisperForConditionalGeneration.from_pretrained(model_name, local_files_only=True)
# processor = WhisperProcessor.from_pretrained(model_name, local_files_only=True)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
processor = AutoProcessor.from_pretrained(model_name)

# Initialize the Whisper ASR pipeline using the loaded model, tokenizer, and feature extractor.
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=device
)

def push_audio(data: bytes):
    """
    Callback function for the Microphone.
    Receives raw PCM bytes and enqueues them for processing.
    """
    # print("DEBUG: Received audio chunk of size", len(data))
    audio_queue.put(data)

def process_audio_buffer(silence_duration: float = 1.5):
    """
    Accumulates audio chunks that pass the energy threshold into an utterance buffer.
    When a period of silence (no above-threshold audio) of at least `silence_duration`
    seconds is detected, the utterance is processed via the ASR pipeline.
    """
    utterance_buffer = bytearray()
    sample_rate = 16000  # Must match the microphone's sample rate.
    last_speech_time = None

    while not stop_event.is_set():
        try:
            data = audio_queue.get(timeout=0.1)
        except queue.Empty:
            # If no new data arrives and we have buffered speech, check for silence timeout.
            if utterance_buffer and last_speech_time is not None:
                if time.time() - last_speech_time >= silence_duration:
                    # print("DEBUG: Silence detected due to timeout. Processing utterance buffer...")
                    _process_buffer(utterance_buffer, sample_rate)
                    utterance_buffer = bytearray()
                    last_speech_time = None
            continue

        # Convert the incoming chunk to a numpy array and compute its RMS energy.
        chunk_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        chunk_energy = np.sqrt(np.mean(chunk_np ** 2))
        # print(f"DEBUG: Chunk RMS energy: {chunk_energy:.5f}")

        if chunk_energy >= energy_threshold:
            # Speech detected: add the chunk to the buffer and update the timestamp.
            utterance_buffer.extend(data)
            last_speech_time = time.time()
        else:
            # Chunk is silent; if we already have buffered speech, check if it's time to process.
            if utterance_buffer and last_speech_time is not None:
                if time.time() - last_speech_time >= silence_duration:
                    # print("DEBUG: Silence detected. Processing utterance buffer...")
                    _process_buffer(utterance_buffer, sample_rate)
                    utterance_buffer = bytearray()
                    last_speech_time = None

    # After stop_event is set, process any remaining buffered audio.
    if utterance_buffer:
        # print("DEBUG: Processing final utterance buffer after stop event...")
        _process_buffer(utterance_buffer, sample_rate)

def _process_buffer(buffer: bytearray, sample_rate: int):
    """
    Helper function to process a complete utterance buffer.
    Converts the raw PCM bytes to a normalized numpy array, computes RMS energy,
    and if above the threshold, transcribes the audio using Whisper.
    """
    audio_np = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0
    energy = np.sqrt(np.mean(audio_np ** 2))
    # print(f"DEBUG: Processed buffer RMS energy = {energy:.5f}")

    if energy > energy_threshold:
        result = asr_pipeline(audio_np)
        transcription = result.get("text", "").strip()
        if transcription:
            # print("Final Transcription:", transcription, "\n")
            _process_transcription(transcription)
        # else:
        #     print("Final Transcription: [No speech detected]\n")
    # else:
    #     print("DEBUG: Processed buffer energy below threshold. Likely silence. No transcription emitted.\n")

#################
# create the OpenAI client
#################
openai_client = openai.OpenAI(
    base_url = "https://agent-88adc820dc97ce2685a2-69qmn.ondigitalocean.app/api/v1/", # subsitute with your own GenAI Agent
    api_key=os.environ.get("DIGITALOCEAN_GENAI_ACCESS_TOKEN_TRAVEL"), # substitute with your own GenAI Agent API Key
)

# initial messages
openai_messages = [
    {
        "role": "assistant",
        "content": "You are a Southern California Travel Agency Assistant. Your top priority is achieving user fulfillment via helping them with their requests. Limit all responses to 2 concise sentences and no more than 100 words.",
    },
]

#################
# Text-to-Speech
#################
kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

def _process_transcription(transcription: str):
    if str(transcription) == "":
        # print("Empty string. Exit.")
        return
    if len(str(transcription)) < 10:
        # print("String is too short to be an answer. Exit.")
        return

    # key word trigger
    if "computer" not in transcription.lower():
        print("No Computer")
        return

    # append messages
    print(f"Speech-to-Text: {transcription}")

    # append the user input to the openai messages
    openai_messages.append(
        {"role": "user", "content": transcription}
    )

    # LLM
    completion = openai_client.chat.completions.create(
        model="n/a",
        messages=openai_messages,
    )

    # result
    save_response = completion.choices[0].message.content

    # print the response
    print(f"\nLLM Response: {save_response}\n")

    # append the response to the openai messages
    openai_messages.append(
        {"role": "assistant", "content": f"{save_response}"}
    )

    # play audio
    samples, sample_rate = kokoro.create(
        save_response, voice="af_sarah", speed=1.0, lang="en-us"
    )
    print("Playing audio...")
    mic.mute()
    sd.play(samples, sample_rate)
    sd.wait()
    mic.unmute()

    # small delay
    time.sleep(0.2)


def main():

    # Start the audio processing thread.
    processing_thread = threading.Thread(target=process_audio_buffer, args=(silence_duration,), daemon=True)
    processing_thread.start()

    # Create a Microphone instance with our push_audio callback.
    global mic
    mic = Microphone(push_callback=push_audio)
    if not mic.start():
        print("Failed to start the microphone.")
        return
    
    print("Recording... Press Enter to stop.")
    input()  # Wait for user input to stop recording.

    # Clean up: stop the microphone and processing thread.
    mic.finish()
    stop_event.set()
    processing_thread.join()
    print("Finished.")


if __name__ == "__main__":
    main()
