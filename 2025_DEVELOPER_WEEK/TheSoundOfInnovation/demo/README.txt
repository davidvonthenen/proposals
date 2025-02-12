For MacOS/Linux assuming you have already installed xcode developer tools, this also requires brew installing for the Microphone capabilities:
- portaudio

Requirements:
1. For using Deepgram's Speech-to-Text, you will need an account which you can sign up for a free at: https://deepgram.com
2. For using ElevenLabs Text-to-Speech (and to clone your voice), you will need an account which you can sign up for a free at: https://elevenlabs.io
3. For using DigitalOcean GenAI Agent (aka RAG Agent Builder Platform), you will need an account which you can sign up at: https://cloud.digitalocean.com/registrations/new

Running the demo:
1. Set your Deepgram API Key to the following environment variable, DEEPGRAM_API_KEY
2. Set your ElevenLabs API Key to the following environment variable, ELEVENLABS_API_KEY
3. Set your DigitalOcean GenAI API Key to the following environment variable, DIGITALOCEAN_GENAI_ACCESS_TOKEN_TRAVEL

I would highly recommend using something like `venv` or `conda`. Then run the following:
1. pip install -r requirements.txt
2. python main.py
