# PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py
# python main.py

from pathlib import Path
import json
import requests
import re
from time import sleep
import os
import random

import torch
import numpy as np
from fastcore.all import *
from fastai.data.all import *
from fastai.vision.all import *

from bs4 import BeautifulSoup
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,  # Import Early Stopping
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

import nltk
from nltk.tokenize import sent_tokenize

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


# Helper functions remain unchanged


def contains_brackets(s):
    brackets = ["{", "}", "[", "]", "(", ")", '"', "?"]
    return any(bracket in s for bracket in brackets)


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
                continue  # skip if sentence contains brackets (or odd charcters) usually Javascript

            non_questions.append(cleaned_sentence)

    # Remove the trailing period from 15% of the sentences
    num_to_modify = int(len(non_questions) * 0.15)
    indices_to_modify = random.sample(range(len(non_questions)), num_to_modify)

    for idx in indices_to_modify:
        if non_questions[idx].endswith("."):
            non_questions[idx] = non_questions[idx][:-1]

    return non_questions


# nltk.download("punkt")

# Load or Create Dataset
df_questions: pd.DataFrame = None
df_non_questions: pd.DataFrame = None
df_combined: pd.DataFrame = None

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

nonquestionPath = Path("non_questions_dataset.csv")
if not nonquestionPath.exists():
    print("Extracting non-questions from Wikipedia.")

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

# Fine-tuning Model
model_path = "question_classifier_model"
tokenizer_path = "question_classifier_tokenizer"

# Fine-tuning the model with device-specific modifications
if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
    if df_combined is None:
        print("Reading combined dataset from CSV.")
        df_combined = pd.read_csv(comboPath)

    # Stratified split
    questions = df_combined[df_combined["is_question"] == 1]
    non_questions = df_combined[df_combined["is_question"] == 0]

    train_questions, test_questions = train_test_split(
        questions, test_size=0.1, random_state=42
    )
    train_non_questions, test_non_questions = train_test_split(
        non_questions, test_size=0.1, random_state=42
    )

    train_combined = pd.concat([train_questions, train_non_questions])
    test_combined = pd.concat([test_questions, test_non_questions])

    ds_train = Dataset.from_pandas(train_combined)
    ds_test = Dataset.from_pandas(test_combined)

    model_nm = "microsoft/deberta-v3-small"
    tokz = AutoTokenizer.from_pretrained(model_nm, force_download=True, use_fast=False)

    def tokenize_function(examples):
        if isinstance(examples["text"], list) and all(
            isinstance(i, str) for i in examples["text"]
        ):
            return tokz(
                examples["text"], padding="max_length", truncation=True, max_length=32
            )
        else:
            raise ValueError(
                "Input 'examples['text']' is not in the correct format. Expected a list of strings."
            )

    tokenized_train = ds_train.map(tokenize_function, batched=True)
    tokenized_test = ds_test.map(tokenize_function, batched=True)

    tokenized_train = tokenized_train.rename_column("is_question", "labels")
    tokenized_test = tokenized_test.rename_column("is_question", "labels")

    tokenized_train = tokenized_train.remove_columns(["text"])
    tokenized_test = tokenized_test.remove_columns(["text"])

    # Load model and send it to the correct device
    model = AutoModelForSequenceClassification.from_pretrained(
        model_nm, num_labels=2
    ).to(device)

    def compute_metrics(p):
        predictions, labels = p
        preds = np.argmax(predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary"
        )
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    # Training arguments with logging, saving, and evaluation strategy
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",  # Ensure evaluation happens every epoch
        save_strategy="epoch",  # Save model every epoch to match evaluation
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,  # Increased number of epochs
        weight_decay=0.01,
        load_best_model_at_end=True,  # Load the best model at the end
        metric_for_best_model="f1",  # Metric to monitor
        logging_dir="./logs",
        logging_steps=10,
        no_cuda=False,  # To ensure CUDA is used if available, but MPS will override
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # Early stopping
    )

    trainer.train()

    # Save the model and tokenizer
    model.save_pretrained("question_classifier_model")
    tokz.save_pretrained("question_classifier_tokenizer")

    print("Model and tokenizer saved successfully.")
else:
    print("Model and tokenizer already exists.")

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
