# BONUS: A Human-Like Sounding Voice Assistant Using All Open Source STT and TTS Components

This is a Voice AI Assistant using open source components for STT and TTS.

## Speech-to-Text: OpenAI whisper-large-v3-turbo

You can either dynamically download the `openai/whisper-large-v3-turbo` model as it currently does in the code (default) OR download the model from huggingface here:

- https://huggingface.co/openai/whisper-large-v3-turbo/tree/main

## Text-to-Speech: Kokoro + onnx

Using Kokoro + onnx. Details:

- https://huggingface.co/spaces/hexgrad/Kokoro-TTS
- https://github.com/thewh1teagle/kokoro-onnx

### For Kokoro + ONNX TTS

Download the following files from their GitHub page:

- kokoro-v1.0.onnx
- voices-v1.0.bin

## Additional Requirements

For MacOS/Linux assuming you have already installed xcode developer tools, this also requires brew installing for the Microphone capabilities:

- portaudio

## Setup and Running the Demo

Requirements:
1. For using DigitalOcean GenAI Agent (aka RAG Agent Builder Platform), you will need an account which you can sign up at: https://cloud.digitalocean.com/registrations/new

Running the demo:
1. Set your DigitalOcean GenAI API Key to the following environment variable, DIGITALOCEAN_GENAI_ACCESS_TOKEN_TRAVEL

I would highly recommend using something like `venv` or `conda`. Then run the following:
1. pip install -r requirements.txt
2. python main.py
