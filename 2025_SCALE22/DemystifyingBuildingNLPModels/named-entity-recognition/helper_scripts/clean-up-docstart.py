import os
import argparse


def clean_conll_file(input_file, output_file):
    """
    Processes a single CoNLL file by removing lines containing '-DOCSTART-' and consolidating empty lines.
    """
    with open(input_file, "r", encoding="utf-8") as infile, open(
        output_file, "w", encoding="utf-8"
    ) as outfile:
        previous_line_empty = False  # Track if the previous line was empty
        for line in infile:
            stripped_line = line.strip()

            # Skip any lines that contain '-DOCSTART-'
            if "-DOCSTART-" in stripped_line:
                continue

            # If current line is empty
            if not stripped_line:
                # If previous line was also empty, skip this line (consolidate)
                if previous_line_empty:
                    continue
                previous_line_empty = True
                outfile.write("\n")
            else:
                previous_line_empty = False
                outfile.write(line)


def process_directory(input_dir, output_dir):
    """
    Recursively processes files in the input directory and saves cleaned versions to the output directory.
    Preserves directory structure.
    """
    for root, dirs, files in os.walk(input_dir):
        # Create corresponding directory structure in the output directory
        relative_path = os.path.relpath(root, input_dir)
        output_subdir = os.path.join(output_dir, relative_path)
        os.makedirs(output_subdir, exist_ok=True)

        # Process each file in the current directory
        for file_name in files:
            input_file = os.path.join(root, file_name)
            output_file = os.path.join(output_subdir, file_name)

            print(f"Processing {input_file} -> {output_file}")
            clean_conll_file(input_file, output_file)


def main(input_dir, output_dir):
    """
    Main function to start the directory processing.
    """
    if not os.path.exists(input_dir):
        print(f"Input directory '{input_dir}' does not exist.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    process_directory(input_dir, output_dir)
    print(f"Processing complete. Cleaned files saved to '{output_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean CoNLL files by removing '-DOCSTART-' lines and consolidating empty lines.")
    parser.add_argument("input_dir", help="Path to the input directory containing CoNLL files.")
    parser.add_argument("output_dir", help="Path to the output directory where cleaned files will be saved.")
    
    args = parser.parse_args()
    
    main(args.input_dir, args.output_dir)
