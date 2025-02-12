import openai
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

from elevenlabs.client import ElevenLabs
from elevenlabs import play

def main():
    try:
        # create the OpenAI client
        client = openai.OpenAI(
            base_url = "https://agent-88adc820dc97ce2685a2-69qmn.ondigitalocean.app/api/v1/",
            api_key=os.environ.get("DIGITALOCEAN_GENAI_ACCESS_TOKEN_TRAVEL"),
        )

        # Deepgram client
        config = DeepgramClientOptions(
            options={"keepalive": "true"},
            # verbose=logging.DEBUG, 
        )
        deepgram: DeepgramClient = DeepgramClient("", config)

        # elevenlabs client
        el_connection = ElevenLabs()

        # create the Microphone
        microphone = Microphone()

        dg_connection = deepgram.listen.websocket.v("1")

        # on_message
        def on_message(self, result, **kwargs):
            # Speech-to-Text
            sentence = result.channel.alternatives[0].transcript
            if len(sentence) < 8:
                return
            if result.is_final is False:
                return

            # key word trigger
            if "Computer" not in sentence:
                print("No Computer")
                return

            print(f"Speech-to-Text: {sentence}")

            # LLM
            try:
                completion = client.chat.completions.create(
                    model="n/a",
                    messages=[
                        {
                            "role": "assistant",
                            "content": "You are a Southern California Travel Agency Assistant. Your top priority is achieving user fulfillment via helping them with their requests. Limit all responses to 2 concise sentences and no more than 75 words.",
                        },
                        {"role": "user", "content": f"{sentence}"},
                    ],
                )
            except Exception as e:
                print(f"LLM Exception: {e}")

            # Text-to-Speech
            llm_reply=completion.choices[0].message.content
            print(f"Text-to-Speech: {llm_reply}")

            try:
                audio = el_connection.text_to_speech.convert(
                    text=llm_reply,
                    voice_id="XXXXXXXX", # you need to pick a model in your inventory. this is custom to my account.
                    model_id="eleven_multilingual_v2",
                    output_format="mp3_44100_128",
                )

                # play the audio
                microphone.mute()
                play(audio)
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
