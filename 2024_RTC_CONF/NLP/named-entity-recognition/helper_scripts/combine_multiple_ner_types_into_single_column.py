import os
import argparse

def process_file(file_path, output_file_path):
    """
    Process a single file, combining columns 2 and 3 using the | character.
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

            # If the line has at least 3 columns (word, type1, type2)
            if len(columns) >= 3:
                # Combine the second and third column using |, but only if they're not the same
                if columns[1] != columns[2]:
                    columns[1] = f"{columns[1]}|{columns[2]}"
                # Remove the third column
                columns.pop(2)

            # Write the updated line back to the file
            outfile.write('\t'.join(columns) + '\n')

def process_directory(input_dir, output_dir):
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
            process_file(input_file_path, output_file_path)

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process NER files and combine type columns.")
    parser.add_argument('input_dir', type=str, help='The input directory containing NER files.')
    parser.add_argument('output_dir', type=str, help='The output directory to save processed files.')

    # Parse the command line arguments
    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Run the directory processing
    process_directory(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
