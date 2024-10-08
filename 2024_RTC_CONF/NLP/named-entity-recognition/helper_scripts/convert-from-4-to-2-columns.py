import os
import argparse

def process_conll_files(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through all files in the input directory
    for filename in os.listdir(input_dir):
        input_file = os.path.join(input_dir, filename)

        # Process only files, not subdirectories
        if os.path.isfile(input_file):
            output_file = os.path.join(output_dir, filename)

            with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
                for line in f_in:
                    # Preserve blank lines (sentence boundary)
                    if line.strip() == '':
                        f_out.write("\n")
                        continue

                    # Split the line into columns using tabs
                    columns = line.strip().split('\t')

                    # Check if the line has at least 4 columns (token, POS, chunk, NER tag)
                    if len(columns) >= 4:
                        # Keep only the 1st (token) and 4th (NER tag) columns
                        new_line = f"{columns[0]}\t{columns[3]}\n"
                        f_out.write(new_line)

    print(f"Processing complete! Cleaned files are saved to: {output_dir}")

# Usage example with command line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CoNLL files from 4 columns to 2 columns.")
    parser.add_argument("input_dir", type=str, help="Path to the input directory containing CoNLL files.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory where cleaned files will be saved.")

    args = parser.parse_args()

    process_conll_files(args.input_dir, args.output_dir)
