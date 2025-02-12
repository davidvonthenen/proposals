import requests
import os

import sounddevice as sd
from pocketsphinx import LiveSpeech
from kokoro_onnx import Kokoro

import openai

# REST API Calls
def send_transcript(role, sentence):
    # get the IP Address from an Environment Variable
    ip_address = os.getenv("DEMO_IP_ADDRESS", "127.0.0.1")
    print(f"Send transcript to server: {ip_address}")

    url = f"http://{ip_address}:3000/transcript"
    data = {
        "role": role,
        "sentence": sentence,
    }
    response = requests.post(url, json=data)

    return response.json()


def main():
    # llama server
    client = openai.OpenAI(
        base_url="http://127.0.0.1:8080/v1",  # "http://<Your api-server IP>:port"
        api_key="sk-no-key-required",
    )

    # Text-to-Speech
    kokoro = Kokoro("kokoro-v0_19.onnx", "voices.bin")

    # Speech-to-Text
    print("Listening for speech...")
    for phrase in LiveSpeech():
        if str(phrase) == "exit":
            break
        if str(phrase) == "":
            continue
        if len(str(phrase)) < 5:
            print("(too short)")
            continue

        print(f"Speech-to-Text: {phrase}")
        send_transcript("user", str(phrase))

        # # key word trigger
        # if "computer" not in phrase:
        #     print("No Computer")
        #     continue

        # LLM
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a Southern California Travel Agency Assistant. Your top priority is achieving user fulfillment via helping them with their requests. Limit all responses to 2 concise sentences and no more than 100 words.",
                },
                {"role": "user", "content": f"{phrase}"},
            ],
        )

        # Text-to-Speech
        response = completion.choices[0].message.content
        # remove "<|im_end|>" from response
        response = response.replace("<|im_end|>", "")
        print(f"Text-to-Speech: {response}")

        # send transcript
        send_transcript("assistant", response)

        # play audio
        samples, sample_rate = kokoro.create(
            response, voice="af_sarah", speed=1.0, lang="en-us"
        )
        print("Playing audio...")
        sd.play(samples, sample_rate)
        sd.wait()


if __name__ == "__main__":
    main()
