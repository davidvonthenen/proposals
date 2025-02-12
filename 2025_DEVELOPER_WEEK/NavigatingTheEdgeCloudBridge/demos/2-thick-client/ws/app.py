import json
import warnings
import os

from flask import Flask, json, request, send_from_directory

# Disable future warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

# REST API
api = Flask(__name__)

@api.route("/transcript", methods=["POST"])
def transcript():
    # get the JSON from the request
    sentence = request.json["sentence"]
    
    # print
    print("\n\n\nTranscript:")
    print(sentence)
    print("\n\n\n")

    # return info in JSON format
    return json.dumps(
        {
            "ack": sentence,
        }
    )

if __name__ == "__main__":
    # get the IP Address from an Environment Variable
    ip_address = os.getenv("DEMO_IP_ADDRESS", "127.0.0.1")

    api.run(host=ip_address, port=3000)
