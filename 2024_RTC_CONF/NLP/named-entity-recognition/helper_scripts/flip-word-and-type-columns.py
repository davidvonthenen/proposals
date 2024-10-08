import os
import sys

def flip_columns(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            # Skip empty lines (sentence boundary)
            if line.strip() == '':
                f_out.write("\n")
                continue

            # Split the line into columns (assuming tab-delimited)
            columns = line.strip().split('\t')

            # Check if there are at least 2 columns to flip
            if len(columns) >= 2:
                # Flip the 1st and 2nd columns
                flipped_line = f"{columns[1]}\t{columns[0]}\n"
                f_out.write(flipped_line)
            else:
                # If there aren't enough columns, write the original line
                f_out.write(line)

    print(f"Processed file saved to: {output_file}")

def process_directory(input_dir, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each file in the input directory
    for filename in os.listdir(input_dir):
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename)
        flip_columns(input_file, output_file)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python flip-word-and-type-columns.py <input_directory> <output_directory>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    process_directory(input_dir, output_dir)
