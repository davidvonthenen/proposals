import warnings

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from captum.attr import LayerIntegratedGradients
import numpy as np
import matplotlib.pyplot as plt
import os

# Disable future warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

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

# Load the trained model and tokenizer
model_path = "question_model"

model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model.to(device).eval()


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

# Forward function for model
def forward_func(inputs, attention_mask=None):
    output = model(inputs, attention_mask=attention_mask)
    logits = output.logits
    return logits


# Helper function to construct input and reference baselines
def construct_input_ref_pair(text, ref_token_id, max_length=32):
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Reference tokens are padding tokens
    ref_input_ids = torch.full_like(input_ids, ref_token_id).to(device)
    return input_ids, ref_input_ids, attention_mask


# Helper function to summarize attributions
def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


# Main function to interpret the model and save visualization
def interpret_and_save_image(text):
    ref_token_id = tokenizer.pad_token_id
    input_ids, ref_input_ids, attention_mask = construct_input_ref_pair(
        text, ref_token_id
    )

    # Initialize LayerIntegratedGradients for embeddings
    lig = LayerIntegratedGradients(forward_func, model.deberta.embeddings)

    # Compute attributions with target class for "Question" (target=1)
    attributions, delta = lig.attribute(
        inputs=input_ids,
        baselines=ref_input_ids,
        additional_forward_args=(attention_mask,),
        target=1,
        return_convergence_delta=True,
    )

    # Summarize attributions
    attr_sum = summarize_attributions(attributions)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Plot the visualization using Matplotlib
    plt.figure(figsize=(10, 5))
    plt.barh(tokens, attr_sum.cpu().detach().numpy(), align="center")
    plt.xlabel("Attribution Score")
    plt.ylabel("Tokens")
    plt.title("Token Attribution Visualization")
    plt.tight_layout()

    # filename
    output_file = text.replace(" ", "_") # replace spaces with underscores
    output_file = "".join(e for e in output_file if e.isalnum() or e == "_") # remove special characters
    output_file = f"{output_file.lower()}.png"

    # Save the visualization
    plt.savefig(output_file)
    print(f"Visualization saved to {output_file}")


# Test the interpretation and save the visualization
if __name__ == "__main__":
    test_sentence = "Tell me about Long Beach, California."

    print("\n\n")
    print(f"Interpreting for: {test_sentence}\n")

    is_question = predict_question(test_sentence)
    print(f"Is '{test_sentence}' a question? {is_question}")
    interpret_and_save_image(test_sentence)

    print("\n\n")
