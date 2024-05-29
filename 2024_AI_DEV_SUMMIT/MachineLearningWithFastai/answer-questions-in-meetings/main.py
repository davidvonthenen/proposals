# run PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
    SpeakOptions,
)

import openai

from fastcore.all import *
from fastai.data.all import *
from fastai.vision.all import *
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from playsound import playsound
from dotenv import load_dotenv

from time import sleep
import os


def main():
    try:
        # openai client
        client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        # question model
        loaded_model = AutoModelForSequenceClassification.from_pretrained(
            "question_classifier_model"
        )
        loaded_tokenizer = AutoTokenizer.from_pretrained(
            "question_classifier_tokenizer", use_fast=False
        )

        # Function to predict if a sentence is a question
        def predict_question(sentence) -> bool:
            inputs = loaded_tokenizer(
                sentence,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            )
            outputs = loaded_model(**inputs)
            logits = outputs.logits
            predicted_class_id = logits.argmax().item()
            return predicted_class_id == 1

        # example of setting up a client config
        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram: DeepgramClient = DeepgramClient("", config)

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

            # sentence
            print(f"Speech-to-Text: {sentence}")

            # is question?
            if predict_question(sentence) is False:
                print("Not a question\n\n")
                return
            print("Is a question\n\n")

            # LLM
            try:
                completion = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are ChatGPT, an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests. Make your responses as concise as possible.",
                        },
                        {"role": "user", "content": f"{sentence}"},
                    ],
                )
            except Exception as e:
                print(f"LLM Exception: {e}")

            # Text-to-Speech
            print(f"Text-to-Speech: {completion.choices[0].message.content}")
            print("\n\n")

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
