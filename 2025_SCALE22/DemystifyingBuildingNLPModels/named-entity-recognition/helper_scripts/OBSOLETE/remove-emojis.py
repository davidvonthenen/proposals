import os
import re

def remove_emojis(text):
    # Emoji removal regex pattern
    emoji_pattern = re.compile(
        "[" 
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"  # other miscellaneous symbols
        u"\U000024C2-\U0001F251"  # enclosed characters
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8', errors='replace') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            # Remove emojis from the line
            cleaned_line = remove_emojis(line)
            f_out.write(cleaned_line)

    print(f"Processed file saved to: {output_file}")

def process_all_files(input_dir, output_dir):
    # Loop through all subdirectories and files in the input directory
    for root, _, files in os.walk(input_dir):
        for filename in files:
            input_file = os.path.join(root, filename)
            # Construct the relative path of the file
            relative_path = os.path.relpath(root, input_dir)
            # Create the corresponding directory structure in the output directory
            new_output_dir = os.path.join(output_dir, relative_path)
            os.makedirs(new_output_dir, exist_ok=True)

            output_file = os.path.join(new_output_dir, filename)

            # Process each file
            process_file(input_file, output_file)

# Usage example:
input_dir = '/path/to/your/input_directory'  # Replace with the path to your input directory
output_dir = '/path/to/your/output_directory'  # Replace with the path to your output directory

process_all_files(input_dir, output_dir)
