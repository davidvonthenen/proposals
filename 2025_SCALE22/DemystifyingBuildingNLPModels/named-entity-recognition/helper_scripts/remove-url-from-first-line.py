import os
import re

# Regular expression pattern to detect a URL
url_pattern = re.compile(r'(http|https)://[^\s]+')

def remove_url_from_first_line(file_path):
    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Check if the first line contains a URL
    if lines and re.search(url_pattern, lines[0]):
        # Remove the first line if it contains a URL
        lines = lines[1:]

        # Write the remaining lines back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        print(f"Removed URL from the first line of: {file_path}")
    else:
        print(f"No URL found in the first line of: {file_path}")

def process_files_in_directory(input_dir):
    # Loop through all files in the directory
    for root, _, files in os.walk(input_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            remove_url_from_first_line(file_path)

# Usage example:
input_dir = 'NER'  # Replace with the path to your directory

process_files_in_directory(input_dir)
