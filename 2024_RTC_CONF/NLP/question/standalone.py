import torch
from fastcore.all import *
from fastai.data.all import *
from fastai.vision.all import *

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


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


def predict_question(sentence):
    # Move the inputs to the device
    inputs = loaded_tokenizer(
        sentence, return_tensors="pt", padding=True, truncation=True, max_length=32
    ).to(device)

    # Move model to the same device
    outputs = loaded_model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    return "Question" if predicted_class_id == 1 else "Not a Question"


# Test predictions
test_sentences = [
    "Is this an example sentence?",
    "My name is John Doe.",
    "The quick brown fox jumps over the lazy dog.",
    "How are you doing my friend",
    "I love to eat pizza because the toppings are delicious.",
    "Tell me about the history of the United States",
]

for sentence in test_sentences:
    print(f"Sentence: '{sentence}' is a '{predict_question(sentence)}'")
