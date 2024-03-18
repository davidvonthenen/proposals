from dotenv import load_dotenv
import openai
from playsound import playsound
from time import sleep
import os

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
    SpeakOptions,
)


def main():
    try:
        client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        # example of setting up a client config. logging values: WARNING, VERBOSE, DEBUG, SPAM
        # config = DeepgramClientOptions(
        #     verbose=logging.DEBUG, options={"keepalive": "true"}
        # )
        # deepgram: DeepgramClient = DeepgramClient("", config)
        # otherwise, use default config
        deepgram: DeepgramClient = DeepgramClient()

        # create the Microphone
        microphone = Microphone()

        dg_connection = deepgram.listen.live.v("1")

        # on_message
        def on_message(self, result, **kwargs):
            # Speech-to-Text
            sentence = result.channel.alternatives[0].transcript
            if len(sentence) < 8:
                return
            if result.is_final is False:
                return

            print(f"Speech-to-Text: {sentence}")

            # LLM
            try:
                completion = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are ChatGPT, an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests.",
                        },
                        {"role": "user", "content": f"{sentence}"},
                    ],
                )
            except Exception as e:
                print(f"LLM Exception: {e}")

            # Text-to-Speech
            print(f"Text-to-Speech: {completion.choices[0].message.content}")

            deepgram = DeepgramClient()

            options = SpeakOptions(
                model="aura-asteria-en",
            )

            SPEAK_OPTIONS = {"text": f"{completion.choices[0].message.content}"}
            try:
                response = deepgram.speak.v("1").save(
                    "output.mp3", SPEAK_OPTIONS, options
                )

                # play the audio
                microphone.mute()
                playsound(f"{response.filename}")
                os.remove(f"{response.filename}")
                sleep(0.5)
                microphone.unmute()
            except Exception as e:
                print(f"TTS Exception: {e}")

        def on_error(self, error, **kwargs):
            print(f"\n\n{error}\n\n")

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(LiveTranscriptionEvents.Error, on_error)

        options: LiveOptions = LiveOptions(
            model="nova-2",
            punctuate=True,
            language="en-US",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
        )
        dg_connection.start(options)

        # set the callback on the microphone object
        microphone.set_callback(dg_connection.send)

        # start microphone
        microphone.start()

        # wait until finished
        input("Press Enter to stop recording...\n\n")

        # Wait for the microphone to close
        microphone.finish()

        # Indicate that we've finished
        dg_connection.finish()

        print("Finished")

    except Exception as e:
        print(f"Exception: {e}")
        return


if __name__ == "__main__":
    main()
