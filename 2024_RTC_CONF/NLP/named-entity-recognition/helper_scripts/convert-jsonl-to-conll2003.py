import json
import os

# Mapping of NER tags
ner_tag_map = {
    0: "O",
    1: "B-PER", 2: "I-PER",
    3: "B-ORG", 4: "I-ORG",
    5: "B-LOC", 6: "I-LOC",
    7: "B-ANIM", 8: "I-ANIM",
    9: "B-BIO", 10: "I-BIO",
    11: "B-CEL", 12: "I-CEL",
    13: "B-DIS", 14: "I-DIS",
    15: "B-EVE", 16: "I-EVE",
    17: "B-FOOD", 18: "I-FOOD",
    19: "B-INST", 20: "I-INST",
    21: "B-MEDIA", 22: "I-MEDIA",
    23: "B-MYTH", 24: "I-MYTH",
    25: "B-PLANT", 26: "I-PLANT",
    27: "B-TIME", 28: "I-TIME",
    29: "B-VEHI", 30: "I-VEHI"
}

def convert_to_conll(jsonl_file, output_file):
    with open(jsonl_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            record = json.loads(line)
            tokens = record.get('tokens', [])
            ner_tags = record.get('ner_tags', [])
            
            for i, token in enumerate(tokens):
                token_text = token
                pos_tag = '-'  # Since POS tag is missing
                chunk_tag = '-'  # Since chunk tag is missing
                
                # Map the NER tag using the provided ner_tag_map
                ner_tag_value = ner_tags[i] if i < len(ner_tags) else 0
                ner_tag = ner_tag_map.get(ner_tag_value, 'O')
                
                # Write in CoNLL format: token POS_tag chunk_tag NER_tag
                f_out.write(f"{token_text} {pos_tag} {chunk_tag} {ner_tag}\n")
            
            # Add a blank line after each sentence
            f_out.write("\n")

def process_folder(input_folder, output_folder):
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.jsonl'):
                input_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                output_file_path = os.path.join(output_dir, file.replace('.jsonl', '_conll.txt'))
                convert_to_conll(input_file_path, output_file_path)

# Usage
input_folder = 'MultiNERD'
output_folder = 'MultiNERD_NEW'

process_folder(input_folder, output_folder)

print(f"Conversion complete! The CoNLL 2003 formatted output is saved to: {output_folder}")
