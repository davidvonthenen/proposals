from pathlib import Path
import warnings
import json

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

# Disable certain warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

# Device setup to prioritize MPS (Apple Silicon), then CUDA, then CPU
device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon) for training.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA (GPU) for training.")
else:
    print("Using CPU for training.")


# Create PyTorch Dataset
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# Training epoch function
def train_epoch(model, data_loader, optimizer, device, scheduler):
    model.train()
    losses = []
    correct_predictions = 0

    for batch in tqdm(data_loader):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()

    # Convert correct_predictions to float32 explicitly for MPS compatibility
    accuracy = correct_predictions.float() / len(data_loader.dataset)
    return accuracy, sum(losses) / len(losses)


# Evaluation function
def eval_model(model, data_loader, device):
    model.eval()
    losses = []
    correct_predictions = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Convert correct_predictions to float32 for MPS
    accuracy = correct_predictions.float() / len(data_loader.dataset)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary"
    )
    return accuracy, sum(losses) / len(losses), f1, precision, recall


# Check if the model is already trained
model_dir = Path("sentiment_model")
if not model_dir.exists():
    # Load the dataset
    data = pd.read_csv("IMDB-Dataset.csv")
    data.rename(columns={"review": "text", "sentiment": "label"}, inplace=True)

    # Encode sentiment labels as integers
    data["label"] = data["label"].map({"positive": 1, "negative": 0})

    # Split the dataset
    train_data, test_data = train_test_split(
        data,
        test_size=0.1,
        random_state=42,
        stratify=data["label"]
    )

    # Initialize tokenizer
    model_name = "microsoft/deberta-v3-small"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        force_download=True, 
        use_fast=False
    )

    # Create Datasets and DataLoaders
    train_dataset = SentimentDataset(
        train_data["text"], train_data["label"], tokenizer
    )
    test_dataset = SentimentDataset(
        test_data["text"], test_data["label"], tokenizer
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2
    ).to(device)

    # Define optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    num_epochs = 5
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # Training loop with early stopping
    best_f1 = 0
    patience = 3
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_acc, train_loss = train_epoch(
            model, train_loader, optimizer, device, scheduler
        )
        print(f"Train loss: {train_loss}, accuracy: {train_acc}")

        val_acc, val_loss, val_f1, val_precision, val_recall = eval_model(
            model, test_loader, device
        )
        print(
            f"Validation loss: {val_loss}, accuracy: {val_acc}, "
            f"F1: {val_f1}, precision: {val_precision}, recall: {val_recall}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save_pretrained("sentiment_model")
            tokenizer.save_pretrained("sentiment_model")
            patience_counter = 0
            print("Saved best model so far.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping.")
                break

# Load the tokenizer and the fine-tuned model
tokenizer = AutoTokenizer.from_pretrained("sentiment_model", use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained("sentiment_model").to(device)

# Prediction function
def predict_sentiment(review):
    model.eval()
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

# Test predictions
test_reviews = [
    "'Misery' was the best movie I've seen since I was a small boy",
    # "'Misery' was the bset moive I've seen snice I was a small boy",
    "'Misery' was the bset moive I've seen snice me were a small boy",
]

print("\n")

for review in test_reviews:
    sentiment, confidence = predict_sentiment(review)
    print(f"Review: '{review}'\nSentiment: {sentiment} (Confidence: {confidence:.2f}%)\n")
