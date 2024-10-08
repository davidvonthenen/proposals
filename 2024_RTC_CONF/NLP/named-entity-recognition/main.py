# PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py

import os
import psutil
from pathlib import Path
import warnings

import torch
import pandas as pd
from transformers import BertTokenizerFast, BertConfig, AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from torch.nn.utils.rnn import pad_sequence
import numpy as np

# Disable future warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

# Function to debug memory usage (not currently used in the code)
def print_memory_usage():
    print(f"Current memory usage: {psutil.virtual_memory().percent}%")


# Set device to GPU if available, otherwise fall back to CPU
# Prioritize Apple Silicon MPS, then CUDA, then CPU
device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon) for training.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA (GPU) for training.")
else:
    print("Using CPU for training.")


# NER tag mapping: Maps NER tags to unique integer indices
ner_tag_map = {
    "B-AFFILIATION": 0,
    "I-AFFILIATION": 1,
    "B-ANATOMICAL": 2,
    "I-ANATOMICAL": 3,
    "B-ATTRIBUTE": 4,
    "I-ATTRIBUTE": 5,
    "B-BRANDS": 6,
    "I-BRANDS": 7,
    "B-DATE": 8,
    "I-DATE": 9,
    "B-DOCUMENT": 10,
    "I-DOCUMENT": 11,
    "B-DRUG": 12,
    "I-DRUG": 13,
    "B-DURATION": 14,
    "I-DURATION": 15,
    "B-EVENT": 16,
    "I-EVENT": 17,
    "B-FAMILY_NAME": 18,
    "I-FAMILY_NAME": 19,
    "B-GIVEN_NAME": 20,
    "I-GIVEN_NAME": 21,
    "B-LOCATION": 22,
    "I-LOCATION": 23,
    "B-MEDICAL-CONDITION": 24,
    "I-MEDICAL-CONDITION": 25,
    "B-MONEY": 26,
    "I-MONEY": 27,
    "B-NAME": 28,
    "I-NAME": 29,
    "B-NUMERIC": 30,
    "I-NUMERIC": 31,
    "B-ORGANIZATION": 32,
    "I-ORGANIZATION": 33,
    "B-OTHER": 34,
    "I-OTHER": 35,
    "B-PRICE": 36,
    "I-PRICE": 37,
    "B-STATUS": 38,
    "I-STATUS": 39,
    "B-TIME": 40,
    "I-TIME": 41,
    "B-MISC": 42,
    "I-MISC": 43,
    "O": 44,
}

# Reverse mapping: Maps indices back to NER tags
ner_tag_map_rev = {v: k for k, v in ner_tag_map.items()}


# Load all data files from the specified directory recursively
def load_all_data_from_directory(directory):
    all_sentences = []
    all_tags = []

    # Traverse directory to find all files
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            print(f"Loading data from {file_path}")
            sentences, tags = load_data(file_path)
            all_sentences.extend(sentences)
            all_tags.extend(tags)

    return all_sentences, all_tags


# Load individual data file in CoNLL format
def load_data(file_path):
    sentences = []
    tags = []
    sentence = []
    tag = []

    with open(file_path, "r") as file:
        for line in file:
            if line.strip() == "":
                if sentence:  # Ensure empty sentences aren't added
                    sentences.append(sentence)
                    tags.append(tag)
                sentence = []
                tag = []
            else:
                token, ner_tag = line.strip().split("\t")
                ner_tags = ner_tag.split("|")  # Split multiple NER tags
                sentence.append(token)
                tag.append(ner_tags)

    # Add the last sentence if file doesn't end with a newline
    if sentence:
        sentences.append(sentence)
        tags.append(tag)

    return sentences, tags


# Tokenization function, aligning tokens with labels
def tokenize_and_preserve_labels(sentences, tags, tokenizer, ner_tag_map):
    tokenized_inputs = tokenizer(
        sentences,
        is_split_into_words=True,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )

    labels = []
    for i, label in enumerate(tags):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(
                    [-100] * len(ner_tag_map)
                )  # Ignored by the loss function
            elif word_idx != previous_word_idx:
                # Start of a new word
                label_vector = [0] * len(ner_tag_map)
                for tag in label[word_idx]:
                    label_vector[ner_tag_map[tag]] = 1
                label_ids.append(label_vector)
            else:
                # Same word as previous (subword token)
                label_vector = [0] * len(ner_tag_map)
                for tag in label[word_idx]:
                    label_vector[ner_tag_map[tag]] = 1
                label_ids.append(label_vector)
            previous_word_idx = word_idx
        labels.append(label_ids)

    return tokenized_inputs, labels


# Custom Dataset class for lazy loading
class NERDataset(Dataset):
    def __init__(self, sentences, tags, tokenizer, ner_tag_map):
        self.sentences = sentences
        self.tags = tags
        self.tokenizer = tokenizer
        self.ner_tag_map = ner_tag_map

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tag = self.tags[idx]

        # Tokenize sentence and preserve labels
        tokenized_inputs, label = tokenize_and_preserve_labels(
            [sentence], [tag], self.tokenizer, self.ner_tag_map
        )

        input_ids = tokenized_inputs["input_ids"][0]
        attention_mask = tokenized_inputs["attention_mask"][0]

        # Convert labels to tensor
        label = torch.tensor(label[0], dtype=torch.float32)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
        }


# Collate function to pads sequences within each batch:
# - Padding: Ensures all sequences in a batch are the same length.
# - Masking: Uses -100 for labels of padded tokens to ignore them during loss computation.
def collate_fn(batch):
    # Extract elements from the batch
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]

    # Pad sequences to the same length
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask_padded = pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )
    labels_padded = pad_sequence(
        labels, batch_first=True, padding_value=-100
    )  # Use -100 to ignore padding in loss computation

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "labels": labels_padded,
    }


# Create DataLoader for training and validation
def create_dataloader(dataset, batch_size=16):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )


# Custom BERT model for multi-label token classification
# - Customization: Adds a linear layer for classification on top of BERT's outputs.
# - Loss Function: Uses BCEWithLogitsLoss suitable for multi-label classification.
class BertForMultiLabelTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForMultiLabelTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)

        # Ensure classifier_dropout has a default value if not in config
        classifier_dropout = getattr(
            config, "classifier_dropout", None
        ) or getattr(  # Get classifier_dropout if available
            config, "hidden_dropout_prob", 0.1
        )  # Default to hidden_dropout_prob or 0.1

        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[
            2:
        ]  # Add hidden states and attention if they are here

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            # Only compute loss on active parts of the labels
            active_loss = labels.view(-1, self.num_labels) != -100
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1, self.num_labels)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


modelPath = Path("ner_model.pth")
if not modelPath.exists():
    # Directory containing the data files
    data_directory = "NER"

    # Load data from the directory
    all_sentences, all_tags = load_all_data_from_directory(data_directory)

    # Split data into train and validation sets (80% train, 20% validation)
    train_sentences, val_sentences, train_tags, val_tags = train_test_split(
        all_sentences, all_tags, test_size=0.2, shuffle=True, random_state=42
    )

    # Initialize BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # Create Dataset objects for training and validation
    train_dataset = NERDataset(train_sentences, train_tags, tokenizer, ner_tag_map)
    val_dataset = NERDataset(val_sentences, val_tags, tokenizer, ner_tag_map)

    # Create DataLoader objects
    train_dataloader = create_dataloader(train_dataset, batch_size=8)
    val_dataloader = create_dataloader(val_dataset, batch_size=8)

    # Define the custom BERT model for multi-label token classification
    config = BertConfig.from_pretrained(
        "bert-base-uncased", num_labels=len(ner_tag_map)
    )
    model = BertForMultiLabelTokenClassification.from_pretrained(
        "bert-base-uncased", config=config
    )
    model.to(device)  # Move model to device

    # Define optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=3e-5)
    total_steps = len(train_dataloader) * 3  # Number of training epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # Training loop
    for epoch in range(3):  # Training for 3 epochs
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            batch_input_ids = batch["input_ids"].to(device)
            batch_attention_masks = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_masks,
                labels=batch_labels,
            )
            loss = outputs[0]
            loss.backward()
            total_loss += loss.item()

            optimizer.step()
            scheduler.step()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} finished. Loss: {avg_loss:.4f}")
    # Evaluate the model
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_dataloader:
            batch_input_ids = batch["input_ids"].to(device)
            batch_attention_masks = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)

            outputs = model(
                input_ids=batch_input_ids, attention_mask=batch_attention_masks
            )
            logits = outputs[0]
            predictions = torch.sigmoid(logits)

            # Apply threshold to get binary predictions
            predictions = (predictions > 0.5).int()

            # Flatten the predictions and labels
            active_loss = batch_labels.view(-1, len(ner_tag_map)) != -100
            active_preds = predictions.view(-1, len(ner_tag_map))[active_loss]
            active_labels = batch_labels.view(-1, len(ner_tag_map))[active_loss]

            all_preds.extend(active_preds.cpu().numpy())
            all_labels.extend(active_labels.cpu().numpy())

    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics
    f1 = f1_score(all_labels, all_preds, average="micro")
    precision = precision_score(all_labels, all_preds, average="micro")
    recall = recall_score(all_labels, all_preds, average="micro")

    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall:    {recall:.4f}")
    print(f"Validation F1 Score:  {f1:.4f}")

    # After training is complete, save the model's state_dict
    torch.save(model.state_dict(), "ner_model.pth")

    # You can also save the entire model if needed (though saving the state_dict is preferred)
    torch.save(model, "ner_model_complete.pth")


# Load the model
config = BertConfig.from_pretrained("bert-base-uncased", num_labels=len(ner_tag_map))
model = BertForMultiLabelTokenClassification(config)
model.load_state_dict(torch.load("ner_model.pth", map_location=device))
model.to(device)
model.eval()


# Function to detect named entities in a sentence
def detect_named_entities(sentence, model, tokenizer, ner_tag_map_rev):
    # Tokenize the input sentence into words
    tokens = sentence.split()

    # Tokenize the input tokens using the tokenizer
    tokenized_inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )

    # Send to device (CPU or GPU)
    tokenized_inputs = {key: val.to(device) for key, val in tokenized_inputs.items()}

    # Predict using the model
    model.eval()
    with torch.no_grad():
        outputs = model(**tokenized_inputs)
        logits = outputs[0]
        predictions = torch.sigmoid(logits)

        # Apply threshold to get binary predictions
        predictions = (predictions > 0.5).int()

    # Convert predictions back to tag names
    predicted_tags = []
    for token_preds in predictions[0]:
        tag_indices = (token_preds == 1).nonzero(as_tuple=True)[0].cpu().numpy()
        tags = [ner_tag_map_rev[idx] for idx in tag_indices]
        predicted_tags.append(tags)

    # Extract original tokens and corresponding predicted labels
    tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"][0])

    # Align tokens and labels (skip special tokens)
    result = []
    for token, tags in zip(tokens, predicted_tags):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        elif token.startswith("##"):
            # Merge subword tokens
            result[-1][0] += token[2:]
        else:
            result.append([token, tags])

    # Merge tokens and their tags
    entities = {}
    for token, tags in result:
        if tags:
            entities[token] = tags

    return entities


# Load the tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# List of test sentences
test_sentences = [
    "Dr. Alice Smith from Stanford University will attend the conference on July 20th.",
    "Google announced a new product in San Francisco last Friday.",
    "John Doe donated $5000 to the Red Cross during the 2020 pandemic.",
    "The Eiffel Tower is one of the most famous landmarks in Paris, France.",
]


# Function to test multiple sentences
def test_ner_model(sentences, model, tokenizer, ner_tag_map_rev):
    for idx, sentence in enumerate(sentences):
        print(f"\nSentence {idx+1}: {sentence}")
        detected_entities = detect_named_entities(
            sentence, model, tokenizer, ner_tag_map_rev
        )
        print("Detected Named Entities:")
        for entity, types in detected_entities.items():
            print(f"  {entity}: {types}")


# Run the test
test_ner_model(test_sentences, model, tokenizer, ner_tag_map_rev)
