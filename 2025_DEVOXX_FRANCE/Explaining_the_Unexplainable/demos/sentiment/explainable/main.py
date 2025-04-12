import warnings
import torch
import numpy as np
import matplotlib.pyplot as plt

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from captum.attr import LayerIntegratedGradients

###############################################################################
# Force PyTorch to use float32 by default for all tensor creations.
###############################################################################
torch.set_default_dtype(torch.float32)

# Disable warnings
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

# Path to your trained sentiment model
model_path = "sentiment_model"

# Load the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model.eval()

def predict_sentiment(review: str):
    """
    Predict the sentiment (positive or negative) of a given review text.
    
    Returns:
        sentiment (str): "positive" or "negative"
        confidence (float): Confidence score (%) of the predicted class
    """
    encoding = tokenizer.encode_plus(
        review,
        add_special_tokens=True,
        max_length=128,
        return_attention_mask=True,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
        predicted_class = logits.argmax(dim=1).item()

    sentiment = "positive" if predicted_class == 1 else "negative"
    confidence = probabilities[0][predicted_class] * 100
    return sentiment, confidence

# Context manager to force torch.tensor calls to use float32.
class ForceTensorFloat32:
    def __enter__(self):
        self.orig_tensor = torch.tensor
        def new_tensor(data, *args, **kwargs):
            # Force dtype to float32 regardless of what's passed.
            kwargs["dtype"] = torch.float32
            return self.orig_tensor(data, *args, **kwargs)
        torch.tensor = new_tensor
    def __exit__(self, exc_type, exc_value, traceback):
        torch.tensor = self.orig_tensor

def interpret_sentiment(review: str):
    """
    Perform LayerIntegratedGradients attribution on the review text,
    targeting the "positive" sentiment class (class index=1).
    
    Returns:
        numpy.ndarray: Normalized attribution scores for each token.
    """
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

    def forward_func(inputs, mask):
        outputs = model(inputs, attention_mask=mask)
        return outputs.logits

    # Instantiate LayerIntegratedGradients for the DeBERTa embeddings.
    lig = LayerIntegratedGradients(forward_func, model.deberta.embeddings)

    # Use the context manager to force tensor creations to be float32.
    with ForceTensorFloat32():
        attributions, delta = lig.attribute(
            inputs=input_ids,
            baselines=torch.zeros_like(input_ids),
            additional_forward_args=(attention_mask,),
            target=1,  # "positive" sentiment class index.
            return_convergence_delta=True,
            n_steps=10,
        )

    # Summarize attributions by summing across the embedding dimensions.
    attributions = attributions.sum(dim=-1).squeeze(0)
    # Normalize by its L2 norm.
    attributions = attributions / torch.norm(attributions)
    # Convert to a CPU NumPy array for visualization.
    attributions = attributions.cpu().detach().numpy()
    return attributions

def visualize_attributions(review: str):
    """
    Visualize token attributions for the given review using matplotlib.
    Also saves a PNG of the visualization.
    """
    sentiment, confidence = predict_sentiment(review)
    attributions = interpret_sentiment(review)

    # Tokenize review text for plotting.
    tokens = tokenizer.tokenize(review)
    max_len = min(len(tokens), len(attributions))
    tokens = tokens[:max_len]
    attr_vals = attributions[:max_len]

    # Create a horizontal bar plot.
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(max_len)
    plt.barh(y_pos, attr_vals, align='center')
    plt.yticks(y_pos, tokens, fontsize=9)
    plt.xlabel("Attribution Score")
    plt.title(f"Token-Level Attribution for '{sentiment}' Sentiment")
    plt.gca().invert_yaxis()  # So the first token is at the top.
    plt.tight_layout()

    # Generate an output filename based on the review text.
    output_file = review.replace(" ", "_").lower()
    output_file = "".join(e for e in output_file if e.isalnum() or e == "_")
    output_file = output_file[:25] + "___.png"
    plt.savefig(output_file)
    plt.close()

    print("\n\n\n")
    print(f"Review: {review}")
    print(f"Predicted sentiment: {sentiment} (confidence: {confidence:.2f}%)")
    print(f"Attributions shape: {attributions.shape}")
    print(f"Visualization saved as '{output_file}'\n")
    print("\n\n\n")

if __name__ == "__main__":
    review_text = "The movie was absolutely fantastic! The story, the characters, everything was perfect."
    visualize_attributions(review_text)

    review_text = "An average experience. Some good moments, but mostly boring."
    visualize_attributions(review_text)
