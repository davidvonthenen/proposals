import torch

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


# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("question_model", use_fast=False)

# Load the fine-tuned model
model = AutoModelForSequenceClassification.from_pretrained("question_model").to(device)


# Define prediction function
def predict_question(sentence):
    model.eval()
    encoding = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=32,
        return_token_type_ids=False,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        _, predicted_class_id = torch.max(logits, dim=1)
        predicted_class_id = predicted_class_id.item()

    return True if predicted_class_id == 1 else False


questionAPI = Flask(__name__)


@questionAPI.route("/question", methods=["POST"])
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
    questionAPI.run(port=3000)
