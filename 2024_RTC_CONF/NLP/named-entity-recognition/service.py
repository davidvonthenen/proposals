import warnings

import torch
from transformers import BertTokenizerFast, BertConfig
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel

from flask import Flask, json, request

# Disable future warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


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


# Custom BERT model for multi-label token classification
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


# Load the model
config = BertConfig.from_pretrained("bert-base-uncased", num_labels=len(ner_tag_map))
model = BertForMultiLabelTokenClassification(config)
model.load_state_dict(torch.load("ner_model.pth", map_location=device))
model.to(device)
model.eval()

# Load the tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


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


def merge_entities(tokens_with_tags):
    entities = []  # List to store the final named entities
    current_entities = {}  # Tracks the current ongoing entities (by tag type)
    current_tokens = {}  # Tracks the words corresponding to each entity type

    for token, tags in tokens_with_tags:
        # For each tag in the list of tags for this token
        for tag in tags:
            if tag == "O":
                # Finalize any ongoing entities before encountering 'O'
                for entity_type, entity_tokens in current_tokens.items():
                    entities.append((" ".join(entity_tokens), entity_type))
                current_tokens = {}
                current_entities = {}
                continue

            if tag.startswith("B-"):
                # Finalize the previous entity if there was one of the same type
                entity_type = tag[2:]
                if entity_type in current_tokens:
                    entities.append((" ".join(entity_tokens), entity_type))

                # Start a new entity of this type
                current_entities[entity_type] = token
                current_tokens[entity_type] = [token]

            elif tag.startswith("I-"):
                entity_type = tag[2:]
                # Continue the existing entity if it matches the type
                if entity_type in current_tokens:
                    current_tokens[entity_type].append(token)
                else:
                    # In case of an 'I-' without a preceding 'B-', treat it as a new entity
                    current_tokens[entity_type] = [token]
                    current_entities[entity_type] = token

    # Finalize any ongoing entities at the end
    for entity_type, entity_tokens in current_tokens.items():
        entities.append((" ".join(entity_tokens), entity_type))

    return entities


api = Flask(__name__)


@api.route("/entity", methods=["POST"])
def entity():
    # get the JSON from the request
    sentence = request.json["sentence"]

    # detect named entities in the sentence
    detected_entities = detect_named_entities(
        sentence, model, tokenizer, ner_tag_map_rev
    )

    # Print the detected entities
    print("Detected Named Entities:")
    for entity, types in detected_entities.items():
        print(f"{entity}: {types}")
    print("\n\n\n")

    # Merge the detected entities
    print("Merged Entities:")
    merged_entities = merge_entities(detected_entities.items())
    for entity_type, entity_text in merged_entities:
        print(f"{entity_type}: {entity_text}")
    print("\n\n\n")

    # return info in JSON format
    return json.dumps(
        {
            "sentence": sentence,
            "detected_entities": detected_entities,
            "merged_entities": merged_entities,
        }
    )


if __name__ == "__main__":
    api.run(port=4000)
