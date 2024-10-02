import warnings
import os
import threading
import requests
import time

from deepgram.utils import verboselogs
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
    SpeakWSOptions,
)

import openai

# Disable future warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

# constants
BOLD = "\033[1m"
RESET = "\033[0m"
MAGENTA = "\033[35m"


def get_entities(sentence):
    url = "http://localhost:4000/entity"
    data = {
        "sentence": sentence,
    }
    response = requests.post(url, json=data)

    return response.json()


def get_question(sentence):
    url = "http://localhost:3000/question"
    data = {
        "sentence": sentence,
    }
    response = requests.post(url, json=data)

    return response.json()


def get_insights(client, entity, type):
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Your name is Elizabeth. You are a helpful assistant. Always limit your responses to using bullet points of the top 3 facts about a subject, but limit the response to 100 words or less.",
                },
                {
                    "role": "user",
                    "content": f"What is {entity} (named entity type: {type})?",
                },
            ],
        )
    except Exception as e:
        print(f"LLM Exception: {e}")

    # Helpful information about an entity
    # print(f"{entity} Description: {completion.choices[0].message.content}\n\n")
    return completion.choices[0].message.content


def main():
    try:
        # openai client
        openai_client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        openai_messages = [
            {
                "role": "system",
                "content": "Your name is Elizabeth. You are a helpful assistant. Always limit your responses to 75 words. Do not go over 75 words. Never say your own name.",
            }
        ]

        # example of setting up a client config
        config = DeepgramClientOptions(
            options={"keepalive": "true", "speaker_playback": "true"},
            # verbose=verboselogs.DEBUG,
        )
        deepgram: DeepgramClient = DeepgramClient("", config)

        # create the Microphone
        microphone = Microphone()

        # listen
        dg_listen_connection = deepgram.listen.websocket.v("1")
        dg_speak_connection = deepgram.speak.websocket.v("1")

        # on_message
        def on_message(self, result, **kwargs):
            # Speech-to-Text
            sentence = result.channel.alternatives[0].transcript
            if len(sentence) < 8:
                return
            if result.is_final is False:
                return

            # sentence
            print(f"Speech-to-Text: {sentence}\n\n")

            if not get_question(sentence)["is_question"]:
                # get entities
                entities_json = get_entities(sentence)

                # get insights
                detected_entities = entities_json["merged_entities"]
                for entity in detected_entities:
                    entity_name, entity_type = entity
                    print(f"Entity Discovered {entity_name} = {entity_type}")
                    insight = get_insights(openai_client, entity_name, entity_type)
                    print(f"{insight}\n\n")
                return
            # else:
            #     print(f"Question DETECTED\n\n")

            # signal to continue displaying insights
            display_event = threading.Event()

            # insights
            user_insights = {}
            llm_insights = {}

            # User discovery thread
            def process_user(sentence):
                # get entities
                entities_json = get_entities(sentence)

                # get insights
                detected_entities = entities_json["merged_entities"]
                for entity in detected_entities:
                    entity_name, entity_type = entity
                    # print(f"Entity Discovered {entity_name} = {entity_type}")
                    if entity_type == "MISC":
                        continue
                    insight = get_insights(openai_client, entity_name, entity_type)
                    user_insights[entity_name + "/" + entity_type] = insight

            # LLM thread
            def process_llm(sentence):
                # append to the openai messages
                openai_messages.append({"role": "user", "content": f"{sentence}"})

                # mute the microphone
                microphone.mute()

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
                                dg_speak_connection.send_text(llm_output)

                    print(f"Text-to-Speech: {save_response}\n\n")
                    openai_messages.append(
                        {"role": "assistant", "content": f"{save_response}"}
                    )
                    dg_speak_connection.flush()

                    # get entities
                    entities_json = get_entities(save_response)

                    # get insights
                    detected_entities = entities_json["merged_entities"]
                    for entity in detected_entities:
                        entity_name, entity_type = entity
                        # print(f"Entity Discovered {entity_name} = {entity_type}")
                        if entity_type == "MISC":
                            continue
                        if entity_type == "BRANDS":
                            continue
                        if entity_type == "EVENT":
                            continue
                        if entity_type == "DATE":
                            continue
                        insight = get_insights(openai_client, entity_name, entity_type)
                        llm_insights[entity_name + "/" + entity_type] = insight

                    # signal to continue displaying insights
                    display_event.set()

                except Exception as e:
                    print(f"LLM Exception: {e}")

            # Start a new thread for processing user
            user_thread = threading.Thread(target=process_user, args=(sentence,))
            user_thread.start()

            # Start a new thread for processing LLM
            llm_thread = threading.Thread(target=process_llm, args=(sentence,))
            llm_thread.start()

            # wait for print insights
            while True:
                time.sleep(0.5)
                if display_event.is_set():
                    total_insights = {**user_insights, **llm_insights}
                    for entity_name_type, insight in total_insights.items():
                        entity_name, entity_type = entity_name_type.split("/")
                        print(f"Entity Discovered {entity_name} = {entity_type}")
                        print(f"Insight:\n{insight}\n\n")
                    break

            # join threads
            user_thread.join()
            llm_thread.join()

            # wait for audio completion
            dg_speak_connection.wait_for_complete()

            # unmute the microphone
            microphone.unmute()

        # def on_error(self, error, **kwargs):
        #     print(f"\n\n{error}\n\n")

        dg_listen_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        # dg_connection.on(LiveTranscriptionEvents.Error, on_error)

        # start speak connection
        speak_options: SpeakWSOptions = SpeakWSOptions(
            model="aura-luna-en",
            encoding="linear16",
            sample_rate=16000,
        )
        dg_speak_connection.start(speak_options)

        # start listen connection
        listen_options: LiveOptions = LiveOptions(
            model="nova-2-conversationalai",
            punctuate=True,
            smart_format=True,
            language="en-US",
            encoding="linear16",
            numerals=True,
            channels=1,
            sample_rate=16000,
            interim_results=True,
            utterance_end_ms="2000",
        )
        dg_listen_connection.start(listen_options)

        # set the callback on the microphone object
        microphone.set_callback(dg_listen_connection.send)

        # start microphone
        microphone.start()

        # wait until finished
        input("\n\n\nPress Enter to stop recording...\n\n")

        # Wait for the microphone to close
        microphone.finish()

        # Indicate that we've finished
        dg_listen_connection.finish()
        dg_speak_connection.finish()

        print("Finished")

    except Exception as e:
        print(f"Exception: {e}")
        return


if __name__ == "__main__":
    main()
