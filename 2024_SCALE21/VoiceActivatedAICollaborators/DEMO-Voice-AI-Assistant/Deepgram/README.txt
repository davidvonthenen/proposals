For MacOS/Linux assuming you have already installed xcode developer tools, this also requires brew installing for the Microphone capabilities:
- portaudio

Requirements:
For using Deepgram's Speech-to-Text, you will need an account which you can sign up for a free at: https://deepgram.com


Deepgram STT + Llama 2 + Deepgram TTS:
------------------------
Steps to run this configuration starting from the same directory as this README.txt.
Will call this repo <ROOT>.
1. [recommended but not required] Create a Python Environment using something like Conda
2. pip install -r requirements.txt
3. git clone git@github.com:ggerganov/llama.cpp.git
4. cd llama.cpp

Inside llama.cpp:
5. mkdir -p ./models/7B
6. download model from page https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF:
    I used the recommended file: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf?download=true
7. mv ./llama-2-7b-chat.Q4_K_M.gguf ./models/7B
8. pip install -r requirements.txt
9. make
10. In a new console, run: ./server -m models/7B/llama-2-7b-chat.Q4_K_M.gguf -c 2048

Go back up to <ROOT>:
11. python deepgram-llama2.py
