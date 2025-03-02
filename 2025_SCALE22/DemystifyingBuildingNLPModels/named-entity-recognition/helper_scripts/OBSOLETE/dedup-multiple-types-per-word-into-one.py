import os
import argparse
import shutil

# Set the input and output directories
input_dir = 'NER'  # Replace with your input directory
output_dir = 'NER-NEW'  # Replace with your output directory

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def process_file(file_path, output_file_path, type1, type2):
    """
    Process a single file, replacing matching combinations of type1 and type2 in columns 2 and 3.
    The result is saved in the output directory while preserving the directory structure.
    """
    with open(file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # Preserve empty lines
            if line.strip() == '':
                outfile.write(line)
                continue

            # Split the line by tabs
            columns = line.strip().split('\t')

            # If the line has at least 3 columns
            if len(columns) >= 3:
                # Check if the 2nd and 3rd column match type1 and type2 in any order
                if (columns[1] == type1 and columns[2] == type2) or (columns[1] == type2 and columns[2] == type1):
                    # Reduce the value to type2 in the second column
                    columns[1] = type2
                    # Remove the third column
                    columns.pop(2)

            # Write the updated line back to the file
            outfile.write('\t'.join(columns) + '\n')

def process_directory(input_dir, output_dir, type1, type2):
    """
    Recursively process all files in the input directory and save modified files to the output directory.
    """
    for root, dirs, files in os.walk(input_dir):
        # Get the relative path to the current directory
        rel_path = os.path.relpath(root, input_dir)
        # Create the corresponding directory in the output directory
        output_root = os.path.join(output_dir, rel_path)
        if not os.path.exists(output_root):
            os.makedirs(output_root)

        # Process each file in the current directory
        for filename in files:
            input_file_path = os.path.join(root, filename)
            output_file_path = os.path.join(output_root, filename)

            # Process the file and write to the output directory
            process_file(input_file_path, output_file_path, type1, type2)

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process NER files and reduce matching type values.")
    parser.add_argument("type1", help="The first type value to match.")
    parser.add_argument("type2", help="The second type value to match. The output will reduce to this value.")
    
    # Parse the command line arguments
    args = parser.parse_args()

    # Run the directory processing with the provided type1 and type2
    process_directory(input_dir, output_dir, args.type1, args.type2)

if __name__ == "__main__":
    main()
