import os
import re
import sys

def print_matches(input_file, match_type, max_matches=5):
    """Print the first column values when the second column matches the given type, case-insensitively."""
    match_count = 0
    with open(input_file, 'r') as infile:
        for line in infile:
            if match_count >= max_matches:
                break  # Stop after reaching the maximum matches
            if line.strip():  # If the line contains content
                # Check if the delimiter is a space and convert to tab
                if re.search(r' \S+', line):  # Checks for a space followed by non-space characters
                    line = re.sub(r' +', '\t', line, count=1)  # Replace spaces with a single tab
                
                columns = line.split('\t')
                if len(columns) == 2:
                    word, label = columns[0], columns[1].strip()
                    if label.lower() == match_type.lower():
                        print(word)
                        match_count += 1

def process_directory(input_dir, match_type, max_matches=5):
    """Recursively process all files in the input directory."""
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            input_file = os.path.join(root, file)
            # print(f"File: {input_file}")
            print_matches(input_file, match_type, max_matches)

if __name__ == "__main__":
    # Ensure match type is passed as the first argument
    if len(sys.argv) < 2:
        print("Please provide a match type as the first parameter.")
        sys.exit(1)

    # Get the match type from the first argument
    match_type = sys.argv[1]

    # Get the max matches from the second argument (if provided), otherwise default to 5
    if len(sys.argv) > 2:
        try:
            max_matches = int(sys.argv[2])  # Convert to integer
        except ValueError:
            print("The second parameter must be an integer (max matches).")
            sys.exit(1)
    else:
        max_matches = 5  # Default value if not provided

    # Define the input directory (can be changed as needed)
    input_directory = 'NER'   # Change this to the path where the files are located

    # Process the directory with the provided match type and max matches
    process_directory(input_directory, match_type, max_matches)
