import json
import os
from pathlib import Path
import re

from flask import Flask, json, request, send_from_directory

import openai


api = Flask(__name__)

@api.route("/recommendation", methods=["POST"])
def recommendation():
    # get the JSON from the request
    sentence = request.json["sentence"]

    # The work... provide the diagnosis
    # OpenAI
    openai_client = openai.OpenAI(
        base_url = "https://agent-88adc820dc97ce2685a2-69qmn.ondigitalocean.app/api/v1/",
        api_key=os.environ.get("DIGITALOCEAN_GENAI_ACCESS_TOKEN_TRAVEL"),
    )
    openai_messages = [
        {
            "role": "assistant",
            "content": "You are a Southern California Travel Agency Assistant. Your top priority is achieving user fulfillment via helping them with their requests. Limit all responses to 2 concise sentences and no more than 100 words.",
        }
    ]

    # append the user input to the openai messages
    openai_messages.append(
        {"role": "user", "content": sentence}
    )

    # send to ChatGPT
    save_response = ""
    try:
        for response in openai_client.chat.completions.create(
            model="n/a",
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

        # print the response
        print(f"\nText-to-Speech: {save_response}\n")

        # append the response to the openai messages
        openai_messages.append(
            {"role": "assistant", "content": f"{save_response}"}
        )

    except Exception as e:
        print("LLM Exception:", e)

    # return info in JSON format
    return json.dumps(
        {
            "reply": save_response,
        }
    )

if __name__ == "__main__":
    # get the IP Address from an Environment Variable
    ip_address = os.getenv("DEMO_IP_ADDRESS", "127.0.0.1")

    api.run(host=ip_address, port=3000)
