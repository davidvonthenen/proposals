import os
from collections import defaultdict, Counter
import nltk

# Ensure NLTK stopwords are downloaded
try:
    from nltk.corpus import stopwords

    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    from nltk.corpus import stopwords

    STOPWORDS = set(stopwords.words("english"))


def read_all_data(directory):
    all_sentences = []
    sentences_by_file = {}
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.startswith("."):
                continue  # Skip hidden files
            file_path = os.path.join(root, filename)
            sentences = read_data(file_path)
            sentences_by_file[file_path] = sentences
            all_sentences.extend(sentences)
    return all_sentences, sentences_by_file


def read_data(file_path):
    sentences = []
    sentence = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip() == "":
                if sentence:
                    sentences.append(sentence)
                    sentence = []
            else:
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    print(f"Skipping malformed line: {line.strip()}")
                    continue
                word, tag_str = parts
                tags = tag_str.split("|")
                token = {"word": word, "tags": tags}
                sentence.append(token)
    if sentence:
        sentences.append(sentence)
    return sentences


def fix_I_tags(sentences):
    for sentence in sentences:
        for idx, token in enumerate(sentence):
            corrected_tags = []
            for tag in token["tags"]:
                if tag.startswith("I-"):
                    entity_type = tag[2:]
                    prev_tags = sentence[idx - 1]["tags"] if idx > 0 else ["O"]
                    prev_entity_types = set(
                        t[2:] for t in prev_tags if t.startswith(("B-", "I-"))
                    )
                    if entity_type not in prev_entity_types:
                        # Incorrect I- tag, should be B-
                        corrected_tag = "B-" + entity_type
                    else:
                        corrected_tag = tag
                    corrected_tags.append(corrected_tag)
                else:
                    corrected_tags.append(tag)
            token["tags"] = corrected_tags
    return sentences


def write_data(file_path, sentences):
    with open(file_path, "w") as f:
        for sentence in sentences:
            for token in sentence:
                tag_str = "|".join(token["tags"])
                f.write(f"{token['word']}\t{tag_str}\n")
            f.write("\n")


def process_files(sentences_by_file):
    for file_path, sentences in sentences_by_file.items():
        print(f"Writing corrected data to file: {file_path}")
        write_data(file_path, sentences)


def identify_inconsistent_entities(all_sentences):
    # Mapping from entity text to tag counts and occurrences
    entity_occurrences = defaultdict(lambda: {"tag_counts": Counter(), "instances": []})
    for sentence_idx, sentence in enumerate(all_sentences):
        for idx, token in enumerate(sentence):
            if STOPWORDS and token["word"].lower() in STOPWORDS:
                continue
            if token["word"].lower() in {"'", '"', ",", ".", ":", ";", "?"}:
                continue

            tags = token["tags"]
            for tag in tags:
                if tag.startswith("B-"):
                    entity_type = tag[2:]

                    if entity_type in {
                        "NAME",
                        "GIVEN_NAME",
                        "FAMILY_NAME",
                        "DATE",
                        "MISC",
                        "OTHER",
                        "O",
                    }:
                        continue

                    entity_tokens = [token["word"]]
                    entity_indices = [idx]
                    j = idx + 1
                    while j < len(sentence):
                        next_token = sentence[j]
                        next_tags = next_token["tags"]
                        matching_tags = [
                            t
                            for t in next_tags
                            if t == "I-" + entity_type or t == "B-" + entity_type
                        ]
                        if matching_tags:
                            entity_tokens.append(next_token["word"])
                            entity_indices.append(j)
                            j += 1
                        else:
                            break
                    entity_text = " ".join(entity_tokens)
                    entity_occurrences[entity_text]["tag_counts"][entity_type] += 1
                    entity_occurrences[entity_text]["instances"].append(
                        {
                            "sentence_idx": sentence_idx,
                            "token_indices": entity_indices,
                            "entity_type": entity_type,
                            "sentence": sentence,
                        }
                    )
    return entity_occurrences


def display_sentence_context(instance, highlight_indices):
    # Reconstruct the sentence as a string, highlighting the entity in question
    sentence = instance["sentence"]
    tokens = []
    for idx, token in enumerate(sentence):
        word = token["word"]
        if idx in highlight_indices:
            # Highlight the word (e.g., using asterisks)
            tokens.append(f"**{word}**")
        else:
            tokens.append(word)
    sentence_str = " ".join(tokens)
    return sentence_str


def prompt_user_decision(
    entity_text, instance, dominant_tag=None, current_prompt=1, total_prompts=1
):
    current_tag = instance["entity_type"]
    sentence_context = display_sentence_context(instance, instance["token_indices"])
    print(f"\nPrompt {current_prompt}/{total_prompts}")
    print("\nSentence:")
    print(sentence_context)
    print(f"Entity: '{entity_text}'")
    if dominant_tag:
        print(f"Dominant tag: '{dominant_tag}'")
    print(f"Current tag: '{current_tag}'")
    print("Options:")
    print("1) Keep it the same (default)")
    print("2) Keep all instances the same")
    print("3) Change this instance to the norm")
    print("4) Change all instances to the norm")
    print(
        "5) Update this instance to given tag(s) (specify old_tag:new_tag1,new_tag2,...)"
    )
    print(
        "6) Update all instances of this entity to given tag(s) (specify old_tag:new_tag1,new_tag2,...)"
    )
    print("7) Replace current value with the dominant value")
    print("s) Save current changes")
    valid_actions = ["1", "2", "3", "4", "5", "6", "7", "s"]
    action = ""
    while action not in valid_actions:
        action = input(f"Choose an action ({'/'.join(valid_actions)}): ").strip()
        if action == "":
            action = "1"

    if action == "1":
        # Keep it the same
        decision = {"action": "keep_same"}
    elif action == "2":
        # Keep all instances the same
        decision = {"action": "keep_all_same"}
    elif action == "3":
        # Change this instance to the norm
        decision = {"action": "change_instance_to_norm", "new_tag": dominant_tag}
    elif action == "4":
        # Change all instances to the norm
        decision = {"action": "change_all_to_norm", "new_tag": dominant_tag}
    elif action == "5":
        # Update this instance to given tag(s)
        tags_input = input(
            "Enter the tag to change and the new tag(s) (old:new1,new2,...): "
        ).strip()
        old_new_tags = tags_input.split(":")
        if len(old_new_tags) != 2:
            print(
                "Invalid input. Please enter in the format old_tag:new_tag1,new_tag2,..."
            )
            return prompt_user_decision(
                entity_text, instance, dominant_tag, current_prompt, total_prompts
            )
        old_tag = old_new_tags[0]
        new_tags = old_new_tags[1].split(",")
        decision = {
            "action": "update_instance_to_given_tag",
            "old_tag": old_tag,
            "new_tags": new_tags,
        }
    elif action == "6":
        # Update all instances of this entity to given tag(s)
        tags_input = input(
            "Enter the tag to change and the new tag(s) (old:new1,new2,...): "
        ).strip()
        old_new_tags = tags_input.split(":")
        if len(old_new_tags) != 2:
            print(
                "Invalid input. Please enter in the format old_tag:new_tag1,new_tag2,..."
            )
            return prompt_user_decision(
                entity_text, instance, dominant_tag, current_prompt, total_prompts
            )
        old_tag = old_new_tags[0]
        new_tags = old_new_tags[1].split(",")
        decision = {
            "action": "update_all_to_given_tag",
            "old_tag": old_tag,
            "new_tags": new_tags,
        }
    elif action == "7":
        # Replace current value with the dominant value
        decision = {"action": "replace_current_with_dominant", "new_tag": dominant_tag}
    elif action == "s":
        # Save current changes
        decision = {"action": "save_progress"}
    else:
        # Keep it the same
        decision = {"action": "keep_same"}

    return decision


def update_instance_tags(instance, old_tag, new_tags):
    # Apply tag changes to this instance
    for idx_in_entity, token_idx in enumerate(instance["token_indices"]):
        token = instance["sentence"][token_idx]
        updated_tags = []
        for tag in token["tags"]:
            if tag[2:] == old_tag:
                # Remove the old tag and add new tags
                prefix = "B-" if tag.startswith("B-") else "I-"
                for new_tag in new_tags:
                    updated_tags.append(prefix + new_tag)
            else:
                # Keep the tag as is
                updated_tags.append(tag)
        token["tags"] = updated_tags


def save_current_changes(all_sentences, sentences_by_file):
    # Update sentences_by_file with current all_sentences
    idx = 0
    for file_path in sentences_by_file:
        num_sentences = len(sentences_by_file[file_path])
        sentences_by_file[file_path] = all_sentences[idx : idx + num_sentences]
        idx += num_sentences
    # Save the changes
    process_files(sentences_by_file)
    print("Progress saved.")


def apply_user_decisions(entity_occurrences, all_sentences, sentences_by_file):
    global_decisions = {}
    total_prompts = calculate_total_prompts(entity_occurrences)
    current_prompt = 1

    for entity_text, data in entity_occurrences.items():
        tag_counts = data["tag_counts"]
        total_occurrences = sum(tag_counts.values())
        dominant_tag, dominant_count = tag_counts.most_common(1)[0]
        dominant_percentage = (dominant_count / total_occurrences) * 100

        # **New Rule Implementation**
        if total_occurrences < 3:
            # Less than 3 instances; assume entity types are correct
            global_decisions[entity_text] = {"action": "keep_all_same"}
            continue  # Skip to next entity

        if dominant_percentage >= 90:
            # Automatically apply option 4: change all instances to the norm
            new_tags = [dominant_tag]
            for inst in data["instances"]:
                update_instance_tags(inst, inst["entity_type"], new_tags)
            # Increment current_prompt by the number of minority instances that we would have prompted
            prompts_skipped = sum(
                1
                for instance in data["instances"]
                if instance["entity_type"] != dominant_tag
            )
            current_prompt += prompts_skipped
            continue  # Skip to next entity
        elif dominant_percentage >= 70:
            # Process minority instances
            instances = data["instances"]
            instances_to_prompt = [
                instance
                for instance in instances
                if instance["entity_type"] != dominant_tag
            ]
        else:
            # Process all instances
            instances = data["instances"]
            instances_to_prompt = instances

        for idx, instance in enumerate(instances_to_prompt):
            decision = prompt_user_decision(
                entity_text,
                instance,
                dominant_tag if dominant_percentage >= 80 else None,
                current_prompt,
                total_prompts,
            )

            # Apply the decision
            action = decision["action"]
            if action == "save_progress":
                # Save current changes
                save_current_changes(all_sentences, sentences_by_file)
                # No change to current_prompt
                continue
            elif action == "keep_same":
                # Do nothing
                current_prompt += 1
            elif action == "keep_all_same":
                prompts_skipped = len(instances_to_prompt) - idx
                current_prompt += prompts_skipped
                global_decisions[entity_text] = decision
                break  # Skip processing further instances
            elif action == "change_instance_to_norm":
                # Change this instance to the dominant tag
                new_tags = [dominant_tag]
                update_instance_tags(instance, instance["entity_type"], new_tags)
                current_prompt += 1
            elif action == "change_all_to_norm":
                # Change all instances to the dominant tag
                new_tags = [dominant_tag]
                for inst in instances[idx:]:
                    update_instance_tags(inst, inst["entity_type"], new_tags)
                prompts_skipped = len(instances_to_prompt) - idx
                current_prompt += prompts_skipped
                global_decisions[entity_text] = decision
                break  # No need to process other instances individually
            elif action == "update_instance_to_given_tag":
                # Update this instance to given tag(s)
                old_tag = decision["old_tag"]
                new_tags = decision["new_tags"]
                update_instance_tags(instance, old_tag, new_tags)
                current_prompt += 1
            elif action == "update_all_to_given_tag":
                # Update all instances to given tag(s)
                old_tag = decision["old_tag"]
                new_tags = decision["new_tags"]
                for inst in instances[idx:]:
                    update_instance_tags(inst, old_tag, new_tags)
                prompts_skipped = len(instances_to_prompt) - idx
                current_prompt += prompts_skipped
                global_decisions[entity_text] = decision
                break  # No need to process other instances individually
            elif action == "replace_current_with_dominant":
                # Replace current value with dominant value
                new_tags = [dominant_tag]
                update_instance_tags(instance, instance["entity_type"], new_tags)
                current_prompt += 1
            else:
                current_prompt += 1


def calculate_total_prompts(entity_occurrences):
    total_prompts = 0
    global_decisions = {}
    for entity_text, data in entity_occurrences.items():
        total_occurrences = sum(data["tag_counts"].values())

        # **New Rule Implementation**
        if total_occurrences < 3:
            # Less than 3 instances; assume entity types are correct
            continue  # No prompts needed

        tag_counts = data["tag_counts"]
        dominant_tag, dominant_count = tag_counts.most_common(1)[0]
        dominant_percentage = (dominant_count / total_occurrences) * 100
        if dominant_percentage >= 90:
            # Automatically apply option 4; no prompts needed
            continue
        elif dominant_percentage >= 80:
            # Count minority instances
            count = sum(
                1
                for instance in data["instances"]
                if instance["entity_type"] != dominant_tag
            )
            total_prompts += count
        else:
            # Count all instances
            total_prompts += len(data["instances"])
    return total_prompts


def main(directory):
    # Read all data
    all_sentences, sentences_by_file = read_all_data(directory)
    # Fix I- tags first
    all_sentences = fix_I_tags(all_sentences)
    # Write back fixed data to files
    idx = 0
    for file_path in sentences_by_file:
        num_sentences = len(sentences_by_file[file_path])
        sentences_by_file[file_path] = all_sentences[idx : idx + num_sentences]
        idx += num_sentences
    # Save the fixed I- tags
    process_files(sentences_by_file)
    # Reload data for inconsistency detection
    all_sentences, sentences_by_file = read_all_data(directory)
    # Identify entity occurrences with tag counts
    entity_occurrences = identify_inconsistent_entities(all_sentences)
    if not entity_occurrences:
        print("No entities found.")
    else:
        # Apply user decisions to correct inconsistencies
        apply_user_decisions(entity_occurrences, all_sentences, sentences_by_file)
    # Update sentences in sentences_by_file
    idx = 0
    for file_path in sentences_by_file:
        num_sentences = len(sentences_by_file[file_path])
        sentences_by_file[file_path] = all_sentences[idx : idx + num_sentences]
        idx += num_sentences
    # Write back to files
    process_files(sentences_by_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fix NER dataset inconsistencies with user interaction."
    )
    parser.add_argument(
        "directory", type=str, help="Directory containing the data files"
    )
    args = parser.parse_args()
    main(args.directory)
