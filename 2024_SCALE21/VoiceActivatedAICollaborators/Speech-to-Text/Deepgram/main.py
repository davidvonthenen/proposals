# Copyright 2023-2024 Deepgram SDK contributors. All Rights Reserved.
# Use of this source code is governed by a MIT license that can be found in the LICENSE file.
# SPDX-License-Identifier: MIT

from dotenv import load_dotenv
import logging, verboselogs
from time import sleep

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    ClientOptionsFromEnv,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)

load_dotenv()


def main():
    try:
        # start the client
        deepgram: DeepgramClient = DeepgramClient(api_key="", config=ClientOptionsFromEnv())

        dg_connection = deepgram.listen.live.v("1")

        def on_message(self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            if len(sentence) == 0:
                return
            print(f"speaker: {sentence}")

        def on_error(self, error, **kwargs):
            print(f"\n\n{error}\n\n")


        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(LiveTranscriptionEvents.Error, on_error)

        options: LiveOptions = LiveOptions(
            model="nova-2",
            smart_format=True,
            language="en-US",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            # To get transcription as it's happening, the following must be set:
            # interim_results=True,
        )
        dg_connection.start(options)

        # Open a microphone stream on the default input device
        microphone = Microphone(dg_connection.send)

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
        print(f"Could not open socket: {e}")
        return


if __name__ == "__main__":
    main()
