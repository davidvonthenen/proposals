import os
import re
import argparse

# Regular expression to detect "Lieutenant" and any modifiers, such as "Lieutenant-General" or "Lieutenant Colonel"
lieutenant_regex = re.compile(
    r"^Lieutenant(?:[-\s]?(Colonel|General|Commander|Junior|Senior))?$", re.IGNORECASE
)
modifier_regex = re.compile(
    r"^(Colonel|General|Commander|Junior|Senior)?$", re.IGNORECASE
)


def process_lieutenant_tokens(outfile, buffer):
    is_lieutenant = (
        False  # Track if we are processing a Lieutenant entity across multiple lines
    )

    for i, (line, columns) in enumerate(buffer):
        token = columns[0]
        ner_tag = columns[1] if len(columns) > 1 else "O"

        # Check if the current token is "Lieutenant" or any modifier
        if lieutenant_regex.match(token):
            if ner_tag in ["B-NAME", "I-NAME", "B-FAMILY_NAME", "I-FAMILY_NAME"]:
                new_tag = f"{ner_tag}|B-ATTRIBUTE"
            else:
                new_tag = "B-ATTRIBUTE"
            is_lieutenant = True  # Set flag to potentially extend attribute tagging on the next line
        elif is_lieutenant and modifier_regex.match(token):
            # If the next token is part of a rank, continue with I-ATTRIBUTE tagging
            new_tag = "I-ATTRIBUTE"
            is_lieutenant = False  # Reset flag after tagging the next rank in sequence
        else:
            # Default behavior for any other tokens, and reset flag
            new_tag = ner_tag
            is_lieutenant = False

        # Write the modified token to the output file
        outfile.write(f"{token}\t{new_tag}\n")


def process_file(file_path, output_file_path):
    with open(file_path, "r", encoding="utf-8") as infile, open(
        output_file_path, "w", encoding="utf-8"
    ) as outfile:
        buffer = []
        for line in infile:
            if line.strip() == "":
                # Sentence boundary; process the buffer and then reset
                process_lieutenant_tokens(outfile, buffer)
                buffer = []
                # Write the empty line to separate sentences
                outfile.write(line)
                continue

            # Split line into token and NER tag
            columns = line.strip().split("\t")
            token_info = (line, columns)
            buffer.append(token_info)

        # Process remaining buffer
        process_lieutenant_tokens(outfile, buffer)


def process_directory(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        # Maintain the directory structure
        rel_path = os.path.relpath(root, input_dir)
        output_root = os.path.join(output_dir, rel_path)
        if not os.path.exists(output_root):
            os.makedirs(output_root)

        # Process each file in the directory
        for filename in files:
            input_file_path = os.path.join(root, filename)
            output_file_path = os.path.join(output_root, filename)
            process_file(input_file_path, output_file_path)


def main():
    parser = argparse.ArgumentParser(
        description="Modify NER tags for 'Lieutenant' and its variations."
    )
    parser.add_argument(
        "input_dir", help="The input directory containing files to process."
    )
    parser.add_argument(
        "output_dir", help="The output directory to save modified files."
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    process_directory(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
