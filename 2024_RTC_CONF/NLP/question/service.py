import torch
from fastcore.all import *
from fastai.data.all import *
from fastai.vision.all import *

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from flask import Flask, json, request


# Device setup to prioritize MPS, then CUDA, and then CPU
device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon) for training.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA (GPU) for training.")
else:
    print("Using CPU for training.")

# Load the model and move it to the appropriate device
loaded_model = AutoModelForSequenceClassification.from_pretrained(
    "question_classifier_model"
).to(device)
loaded_tokenizer = AutoTokenizer.from_pretrained(
    "question_classifier_tokenizer", use_fast=False
)


def predict_question(sentence) -> bool:
    # Move the inputs to the device
    inputs = loaded_tokenizer(
        sentence, return_tensors="pt", padding=True, truncation=True, max_length=32
    ).to(device)

    # Move model to the same device
    outputs = loaded_model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    return True if predicted_class_id == 1 else False


api = Flask(__name__)


@api.route("/question", methods=["POST"])
def question():
    # get the JSON from the request
    sentence = request.json["sentence"]

    # predict if a question
    is_question = predict_question(sentence)
    print(f"Is '{sentence}' a question? {is_question}")

    # return info in JSON format
    return json.dumps(
        {
            "sentence": sentence,
            "is_question": is_question,
        }
    )


if __name__ == "__main__":
    api.run(port=3000)
