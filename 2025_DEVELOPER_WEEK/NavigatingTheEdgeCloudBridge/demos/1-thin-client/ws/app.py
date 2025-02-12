import json
import multiprocessing
import time
import os
from pathlib import Path
import re

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
        # base_url="http://localhost:8080/v1",
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    openai_messages = [
        {
            "role": "system",
            "content": "You are a Southern California Travel Agency Assistant. Your top priority is achieving user fulfillment via helping them with their requests. Limit all responses to 2 concise sentences and no more than 100 words.",
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

        # LLM
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

def run_ws():
    # get the IP Address from an Environment Variable
    ip_address = os.getenv("DEMO_IP_ADDRESS", "127.0.0.1")

    with serve(hello, ip_address, 3000) as server:
        server.serve_forever()

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    p_ws = multiprocessing.Process(target=run_ws)
    p_ws.start()
    p_ws.join()
