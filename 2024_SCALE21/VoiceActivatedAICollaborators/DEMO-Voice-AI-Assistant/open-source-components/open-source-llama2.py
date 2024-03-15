import pyttsx3
from pocketsphinx import LiveSpeech
import openai


def main():
    client = openai.OpenAI(
        base_url="http://localhost:8080/v1",  # "http://<Your api-server IP>:port"
        api_key="sk-no-key-required",
    )

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

        # LLM
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are ChatGPT, an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests.",
                },
                {"role": "user", "content": f"{phrase}"},
            ],
        )

        # Text-to-Speech
        print(f"Text-to-Speech: {completion.choices[0].message.content}")

        engine = pyttsx3.init()
        engine.say(completion.choices[0].message.content)
        engine.runAndWait()


if __name__ == "__main__":
    main()
