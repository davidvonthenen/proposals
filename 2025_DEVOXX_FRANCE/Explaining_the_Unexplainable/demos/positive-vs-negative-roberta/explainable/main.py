#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example script for sentiment classification and attribution using the CardiffNLP Twitter RoBERTa model.

Model reference:
https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
"""
import warnings
import logging

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import LayerIntegratedGradients

# Disable future warnings and excessive logging
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# Mapping from class indices to sentiment labels (CardiffNLP model):
# 0 -> Negative, 1 -> Neutral, 2 -> Positive.
LABEL_MAPPING = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Specify the path to the local model directory.
model_dir = "/Users/dvonthenen/models/twitter-roberta-base-sentiment"  # Adjust if needed.

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

def predict_sentiment(texts, tokenizer, model, device):
    """
    Predict the sentiment of a list of texts using the provided model and tokenizer.

    Args:
        texts (list of str): List of input texts to classify.
        tokenizer: The HuggingFace tokenizer.
        model: The sentiment classification PyTorch model.
        device: The device to run inference on (CPU or CUDA).

    Returns:
        list of tuples: (input text, predicted sentiment label, probability mapping).
    """
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: tensor.to(device) for key, tensor in inputs.items()}

    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)

    # Apply softmax to obtain probability distribution over sentiment classes
    probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()
    predictions = probs.argmax(axis=1)

    results = []
    for text, pred, prob_array in zip(texts, predictions, probs):
        label = LABEL_MAPPING.get(pred, f"Label-{pred}")
        prob_mapping = {
            "Negative": float(prob_array[0]),
            "Neutral": float(prob_array[1]),
            "Positive": float(prob_array[2]),
        }
        results.append((text, label, prob_mapping))
    return results

def single_text_prediction(text, tokenizer, model, device):
    """
    Wrapper to handle a single text instead of a list.
    Returns the predicted class index and the confidence.
    """
    results = predict_sentiment([text], tokenizer, model, device)
    # We only passed one text, so unpack the single tuple
    _, label, prob_mapping = results[0]

    # Extract the numeric index of the predicted label by reversing LABEL_MAPPING
    label_to_idx = {v: k for k, v in LABEL_MAPPING.items()}
    predicted_idx = label_to_idx[label]

    # Confidence is the probability of the predicted label
    confidence = prob_mapping[label]
    return predicted_idx, confidence, label

def interpret_sentiment(review, tokenizer, model, device):
    """
    Perform LayerIntegratedGradients attribution on the review text,
    targeting the "Positive" sentiment class (index=2) for the CardiffNLP model.

    Returns:
        numpy.ndarray: normalized attribution scores for each token.
    """
    # Encode text
    encoding = tokenizer.encode_plus(
        review,
        max_length=128,
        return_attention_mask=True,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Define forward function for IG
    def forward_func(inputs, mask):
        outputs = model(inputs, attention_mask=mask)
        return outputs.logits

    # CardiffNLP Twitter RoBERTa uses `model.roberta.embeddings` for embeddings
    lig = LayerIntegratedGradients(forward_func, model.roberta.embeddings)

    # We'll interpret for the 'Positive' class, which is index=2
    attributions, delta = lig.attribute(
        inputs=input_ids,
        baselines=torch.zeros_like(input_ids),  # Zero baseline
        additional_forward_args=(attention_mask,),
        target=2,
        return_convergence_delta=True,
    )

    # Sum across embedding dimensions
    attributions = attributions.sum(dim=-1).squeeze(0)
    # Normalize by its L2 norm
    attributions = attributions / torch.norm(attributions)
    # Convert to CPU numpy
    attributions = attributions.detach().cpu().numpy()

    return attributions

def visualize_attributions(sentiment, review, tokenizer, model, device):
    """
    Visualize token-level attributions for the given review using matplotlib.

    Saves the resulting bar chart as a PNG file named after the truncated review text.
    """
    # Get sentiment prediction
    predicted_idx, confidence, label = single_text_prediction(review, tokenizer, model, device)

    # Perform integrated gradients for the 'Positive' class (index=2)
    attributions = interpret_sentiment(review, tokenizer, model, device)

    # For illustration, do a naive tokenization of the input text
    tokens = tokenizer.tokenize(review)

    # Align tokens with attributions
    max_len = min(len(tokens), len(attributions))
    tokens = tokens[:max_len]
    attr_vals = attributions[:max_len]

    # Create a horizontal bar plot
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(max_len)
    plt.barh(y_pos, attr_vals, align="center")
    plt.yticks(y_pos, tokens, fontsize=9)
    plt.xlabel("Attribution Score")
    plt.title(f"Token-Level Attribution for '{label}' Sentiment")

    # Invert y-axis so the first token appears at the top
    plt.gca().invert_yaxis()
    plt.tight_layout()

    # Build a filename from the review text
    output_file = f"{sentiment}-{review}".replace(" ", "_").lower()
    output_file = "".join(e for e in output_file if e.isalnum() or e == "_" or e == "-")
    output_file = output_file[:25] + "___.png"

    plt.savefig(output_file)
    plt.close()

    # Print out the key info
    print("\n\n")
    print(f"Review: {review}")
    print(f"Predicted Sentiment: {label} (confidence: {confidence * 100:.2f}%)")
    print(f"Attributions shape: {attributions.shape}")
    print(f"Visualization saved as '{output_file}'")
    print("\n\n")

if __name__ == "__main__":
    # Example texts
    texts = [
        "'Misery' was the best movie I've seen since I was a small boy",
        "'Misery' was the bset moive I've seen snice I was a small boy",
        "'Misery' was the bset moive I've seen snice me were a small boy",
    ]

    # Determine whether to run on GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Run a batch sentiment prediction for demonstration
    results = predict_sentiment(texts, tokenizer, model, device)
    for text, label, prob_mapping in results:
        print(f"Text: {text}")
        print(f"Predicted Sentiment: {label}")
        for sentiment, probability in prob_mapping.items():
            print(f"  {sentiment}: {probability:.4f}")

        visualize_attributions(label, text, tokenizer, model, device)
