from pathlib import Path
import json
import requests
import os
import random
import warnings

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from bs4 import BeautifulSoup
import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from tqdm import tqdm

import nltk
from nltk.tokenize import sent_tokenize

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

# Load or Create Dataset
df_questions: pd.DataFrame = None
df_non_questions: pd.DataFrame = None
df_combined: pd.DataFrame = None

print("Step 1: Done!")


# Download SQuAD Dataset from Stanford and Load or Create Questions Dataset
def remove_question_marks(df, percentage=0.15):
    np.random.seed(42)  # For reproducibility
    mask = df["is_question"] == 1
    question_indices = df[mask].index
    num_to_modify = int(len(question_indices) * percentage)
    indices_to_modify = np.random.choice(question_indices, num_to_modify, replace=False)
    df.loc[indices_to_modify, "text"] = df.loc[indices_to_modify, "text"].str.rstrip(
        "?"
    )
    return df


questionPath = Path("questions_dataset.csv")
if not questionPath.exists():
    print("Extracting questions from SQuAD dataset.")

    local_filename = "train-v2.0.json"

    if not os.path.exists(local_filename):
        url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"

        response = requests.get(url)
        response.raise_for_status()

        with open(local_filename, "wb") as f:
            f.write(response.content)

    with open(local_filename) as f:
        squad_data = json.load(f)

    questions = []
    for article in squad_data["data"]:
        for paragraph in article["paragraphs"]:
            for qas in paragraph["qas"]:
                questions.append(qas["question"])

    tmp = pd.DataFrame(questions, columns=["text"])
    tmp["is_question"] = 1

    df_questions = remove_question_marks(tmp, 0.15)

    df_questions.to_csv("questions_dataset.csv", index=False)
    print("CSV file 'questions_dataset.csv' created successfully.")
else:
    print("df_questions already exists.")

print("Step 2: Done!")


# Load or Create Non-Questions Dataset
def contains_brackets(s):
    brackets = ["{", "}", "[", "]", "(", ")", '"', "?"]
    return any(bracket in s for bracket in brackets)


def extract_non_questions_from_wikipedia(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    paragraphs = soup.find_all("p")
    non_questions = []

    for para in paragraphs:
        sentences = sent_tokenize(para.get_text())
        for sentence in sentences:
            cleaned_sentence = sentence.strip()
            if len(cleaned_sentence) < 10:
                continue  # skip if sentence is too short
            if contains_brackets(cleaned_sentence):
                continue  # skip if sentence contains brackets (or odd characters)

            non_questions.append(cleaned_sentence)

    # Remove the trailing period from 15% of the sentences
    num_to_modify = int(len(non_questions) * 0.15)
    indices_to_modify = random.sample(range(len(non_questions)), num_to_modify)

    for idx in indices_to_modify:
        if non_questions[idx].endswith("."):
            non_questions[idx] = non_questions[idx][:-1]

    return non_questions


nonquestionPath = Path("non_questions_dataset.csv")
if not nonquestionPath.exists():
    print("Extracting non-questions from Wikipedia.")

    nltk.download("punkt")

    urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://en.wikipedia.org/wiki/Natural_language_processing",
        "https://en.wikipedia.org/wiki/Computer_vision",
        "https://en.wikipedia.org/wiki/Deep_learning",
        "https://en.wikipedia.org/wiki/Artificial_neural_network",
        "https://en.wikipedia.org/wiki/Data_science",
        "https://en.wikipedia.org/wiki/Big_data",
        "https://en.wikipedia.org/wiki/Quantum_computing",
        "https://en.wikipedia.org/wiki/Cryptography",
        "https://en.wikipedia.org/wiki/Computer_security",
        "https://en.wikipedia.org/wiki/Internet_of_things",
        "https://en.wikipedia.org/wiki/Blockchain",
        "https://en.wikipedia.org/wiki/Cloud_computing",
        "https://en.wikipedia.org/wiki/Augmented_reality",
        "https://en.wikipedia.org/wiki/Virtual_reality",
        "https://en.wikipedia.org/wiki/Robotics",
        "https://en.wikipedia.org/wiki/Automation",
        "https://en.wikipedia.org/wiki/Autonomous_car",
        "https://en.wikipedia.org/wiki/Space_exploration",
        "https://en.wikipedia.org/wiki/Renewable_energy",
        "https://en.wikipedia.org/wiki/Climate_change",
        "https://en.wikipedia.org/wiki/Sustainable_development",
        "https://en.wikipedia.org/wiki/Genetics",
        "https://en.wikipedia.org/wiki/Biotechnology",
        "https://en.wikipedia.org/wiki/Nanotechnology",
        "https://en.wikipedia.org/wiki/Astrophysics",
        "https://en.wikipedia.org/wiki/Theoretical_physics",
        "https://en.wikipedia.org/wiki/Artificial_general_intelligence",
        "https://en.wikipedia.org/wiki/Philosophy_of_mind",
        "https://en.wikipedia.org/wiki/History_of_computing",
        "https://en.wikipedia.org/wiki/Information_theory",
        "https://en.wikipedia.org/wiki/Software_engineering",
        "https://en.wikipedia.org/wiki/Algorithm",
        "https://en.wikipedia.org/wiki/Data_structure",
        "https://en.wikipedia.org/wiki/Operating_system",
        "https://en.wikipedia.org/wiki/Computer_graphics",
        "https://en.wikipedia.org/wiki/Parallel_computing",
        "https://en.wikipedia.org/wiki/Distributed_computing",
        "https://en.wikipedia.org/wiki/Database_management_system",
        "https://en.wikipedia.org/wiki/Compiler",
        "https://en.wikipedia.org/wiki/Artificial_life",
        "https://en.wikipedia.org/wiki/Bioinformatics",
        "https://en.wikipedia.org/wiki/Computational_biology",
        "https://en.wikipedia.org/wiki/Digital_signal_processing",
        "https://en.wikipedia.org/wiki/Computer_network",
        "https://en.wikipedia.org/wiki/Wireless_communication",
        "https://en.wikipedia.org/wiki/History_of_the_Internet",
        "https://en.wikipedia.org/wiki/Evolutionary_computation",
        "https://en.wikipedia.org/wiki/History_of_computer_science",
        "https://en.wikipedia.org/wiki/Software",
        "https://en.wikipedia.org/wiki/Programming_language",
        "https://en.wikipedia.org/wiki/Software_development",
        "https://en.wikipedia.org/wiki/Operating_system",
        "https://en.wikipedia.org/wiki/Computer_program",
        "https://en.wikipedia.org/wiki/Computer_programming",
        "https://en.wikipedia.org/wiki/Algorithm",
        "https://en.wikipedia.org/wiki/Data_structure",
        "https://en.wikipedia.org/wiki/Database",
        "https://en.wikipedia.org/wiki/Relational_database",
        "https://en.wikipedia.org/wiki/SQL",
        "https://en.wikipedia.org/wiki/NoSQL",
        "https://en.wikipedia.org/wiki/Distributed_database",
        "https://en.wikipedia.org/wiki/Computer_network",
        "https://en.wikipedia.org/wiki/Internet",
        "https://en.wikipedia.org/wiki/World_Wide_Web",
        "https://en.wikipedia.org/wiki/Hypertext_Transfer_Protocol",
        "https://en.wikipedia.org/wiki/Hypertext_Markup_Language",
        "https://en.wikipedia.org/wiki/Cascading_Style_Sheets",
        "https://en.wikipedia.org/wiki/JavaScript",
        "https://en.wikipedia.org/wiki/Web_browser",
        "https://en.wikipedia.org/wiki/Cloud_storage",
        "https://en.wikipedia.org/wiki/Virtual_machine",
        "https://en.wikipedia.org/wiki/Containerization_(computing)",
        "https://en.wikipedia.org/wiki/Microservices",
        "https://en.wikipedia.org/wiki/DevOps",
        "https://en.wikipedia.org/wiki/Agile_software_development",
        "https://en.wikipedia.org/wiki/Scrum_(software_development)",
        "https://en.wikipedia.org/wiki/Kanban",
        "https://en.wikipedia.org/wiki/Lean_software_development",
        "https://en.wikipedia.org/wiki/Extreme_programming",
        "https://en.wikipedia.org/wiki/Software_testing",
        "https://en.wikipedia.org/wiki/Unit_testing",
        "https://en.wikipedia.org/wiki/Integration_testing",
        "https://en.wikipedia.org/wiki/System_testing",
        "https://en.wikipedia.org/wiki/Acceptance_testing",
        "https://en.wikipedia.org/wiki/Regression_testing",
        "https://en.wikipedia.org/wiki/Performance_testing",
        "https://en.wikipedia.org/wiki/Load_testing",
        "https://en.wikipedia.org/wiki/Stress_testing_(software)",
        "https://en.wikipedia.org/wiki/Usability_testing",
        "https://en.wikipedia.org/wiki/Security_testing",
        "https://en.wikipedia.org/wiki/Software_maintenance",
        "https://en.wikipedia.org/wiki/Software_deployment",
        "https://en.wikipedia.org/wiki/Version_control",
        "https://en.wikipedia.org/wiki/Git",
        "https://en.wikipedia.org/wiki/Mercurial",
        "https://en.wikipedia.org/wiki/Subversion_(software)",
        "https://en.wikipedia.org/wiki/Continuous_integration",
        "https://en.wikipedia.org/wiki/Continuous_delivery",
        "https://en.wikipedia.org/wiki/Continuous_deployment",
        "https://en.wikipedia.org/wiki/Infrastructure_as_Code",
        "https://en.wikipedia.org/wiki/Platform_as_a_service",
        "https://en.wikipedia.org/wiki/Software_as_a_service",
        "https://en.wikipedia.org/wiki/Infrastructure_as_a_service",
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://en.wikipedia.org/wiki/Deep_learning",
        "https://en.wikipedia.org/wiki/Neural_network",
        "https://en.wikipedia.org/wiki/Convolutional_neural_network",
        "https://en.wikipedia.org/wiki/Recurrent_neural_network",
        "https://en.wikipedia.org/wiki/Generative_adversarial_network",
        "https://en.wikipedia.org/wiki/Support_vector_machine",
        "https://en.wikipedia.org/wiki/Decision_tree",
        "https://en.wikipedia.org/wiki/Random_forest",
        "https://en.wikipedia.org/wiki/K-means_clustering",
        "https://en.wikipedia.org/wiki/Principal_component_analysis",
        "https://en.wikipedia.org/wiki/Linear_regression",
        "https://en.wikipedia.org/wiki/Logistic_regression",
        "https://en.wikipedia.org/wiki/Gradient_descent",
        "https://en.wikipedia.org/wiki/Backpropagation",
        "https://en.wikipedia.org/wiki/Stochastic_gradient_descent",
        "https://en.wikipedia.org/wiki/Reinforcement_learning",
        "https://en.wikipedia.org/wiki/Markov_decision_process",
        "https://en.wikipedia.org/wiki/Q-learning",
        "https://en.wikipedia.org/wiki/Monte_Carlo_tree_search",
        "https://en.wikipedia.org/wiki/Bayesian_network",
        "https://en.wikipedia.org/wiki/Hidden_Markov_model",
        "https://en.wikipedia.org/wiki/Naive_Bayes_classifier",
        "https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm",
        "https://en.wikipedia.org/wiki/Dimensionality_reduction",
        "https://en.wikipedia.org/wiki/Feature_engineering",
        "https://en.wikipedia.org/wiki/Anomaly_detection",
        "https://en.wikipedia.org/wiki/Ensemble_learning",
        "https://en.wikipedia.org/wiki/Transfer_learning",
        "https://en.wikipedia.org/wiki/Federated_learning",
        "https://en.wikipedia.org/wiki/Active_learning_(machine_learning)",
        "https://en.wikipedia.org/wiki/Semi-supervised_learning",
        "https://en.wikipedia.org/wiki/Unsupervised_learning",
        "https://en.wikipedia.org/wiki/Supervised_learning",
        "https://en.wikipedia.org/wiki/Recommender_system",
        "https://en.wikipedia.org/wiki/Collaborative_filtering",
        "https://en.wikipedia.org/wiki/Content-based_filtering",
        "https://en.wikipedia.org/wiki/Hybrid_recommender_system",
        "https://en.wikipedia.org/wiki/Contextual_bandit",
        "https://en.wikipedia.org/wiki/A/B_testing",
        "https://en.wikipedia.org/wiki/Multivariate_testing",
        "https://en.wikipedia.org/wiki/Exploratory_data_analysis",
        "https://en.wikipedia.org/wiki/Confirmatory_data_analysis",
    ]

    non_questions = []
    for url in urls:
        if len(non_questions) == 0:
            non_questions = extract_non_questions_from_wikipedia(url)
        else:
            tmp = extract_non_questions_from_wikipedia(url)
            non_questions.extend(tmp)

    df_non_questions = pd.DataFrame(non_questions, columns=["text"])
    df_non_questions["is_question"] = 0

    df_non_questions.to_csv("non_questions_dataset.csv", index=False)
    print("CSV file 'non_questions_dataset.csv' created successfully.")
else:
    print("df_non_questions already exists.")

print("Step 3: Done!")

# Combine DataFrames
comboPath = Path("df_combined_dataset.csv")
if not comboPath.exists():
    print("Combining questions and non-questions into one dataset.")
    if df_questions is None:
        df_questions = pd.read_csv(questionPath)
    if df_non_questions is None:
        df_non_questions = pd.read_csv(nonquestionPath)

    df_combined = pd.concat([df_questions, df_non_questions], ignore_index=True)
    df_combined.to_csv("df_combined_dataset.csv", index=False)
    print("CSV file 'df_combined_dataset.csv' created successfully.")
else:
    print("df_combined_dataset already exists.")

print("Step 4: Done!")

# Train the Model
# Create PyTorch Dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=32):
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
        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }
        return item


# Training Functions
def train_epoch(model, data_loader, optimizer, device, scheduler):
    model.train()
    losses = []
    correct_predictions = torch.tensor(0.0, device=device)

    for batch in tqdm(data_loader):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)
        correct_predictions += (preds == labels).sum().float()

        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()

    accuracy = correct_predictions.float() / len(data_loader.dataset)
    return accuracy.cpu().item(), np.mean(losses)


# Evaluation Functions
def eval_model(model, data_loader, device):
    model.eval()
    losses = []
    correct_predictions = torch.tensor(0.0, device=device)
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)
            correct_predictions += (preds == labels).sum().float()

            losses.append(loss.item())

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = correct_predictions.float() / len(data_loader.dataset)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary"
    )
    return accuracy.cpu().item(), np.mean(losses), f1, precision, recall


modelDir = Path("question_model")
if not modelDir.exists():
    # Prepare Data for PyTorch
    if df_combined is None:
        print("Reading combined dataset from CSV.")
        df_combined = pd.read_csv(comboPath)

    # Stratified split
    train_df, test_df = train_test_split(
        df_combined, test_size=0.1, random_state=42, stratify=df_combined["is_question"]
    )

    # Initialize tokenizer
    model_nm = "microsoft/deberta-v3-small"
    tokenizer = AutoTokenizer.from_pretrained(
        model_nm, force_download=True, use_fast=False
    )

    train_dataset = TextDataset(
        texts=train_df["text"],
        labels=train_df["is_question"],
        tokenizer=tokenizer,
        max_length=32,
    )

    test_dataset = TextDataset(
        texts=test_df["text"],
        labels=test_df["is_question"],
        tokenizer=tokenizer,
        max_length=32,
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Load model and send it to the correct device
    model = AutoModelForSequenceClassification.from_pretrained(
        model_nm, num_labels=2
    ).to(device)

    # Define optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    num_epochs = 5
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    criterion = torch.nn.CrossEntropyLoss()

    # Training Loop with Early Stopping
    best_f1 = 0
    patience = 3
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_acc, train_loss = train_epoch(
            model, train_loader, optimizer, device, scheduler
        )
        print(f"Train loss {train_loss} accuracy {train_acc}")

        val_acc, val_loss, val_f1, val_precision, val_recall = eval_model(
            model, test_loader, device
        )
        print(f"Val   loss {val_loss} accuracy {val_acc} f1 {val_f1}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            # Save the model and tokenizer
            # torch.save(model.state_dict(), "best_model_state.bin")
            model.save_pretrained("question_model")
            tokenizer.save_pretrained("question_model")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break
else:
    print("Model already exists. Loading the model.")

print("Step 5: Done!")

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
