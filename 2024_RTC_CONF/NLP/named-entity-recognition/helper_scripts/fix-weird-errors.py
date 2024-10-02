import os
import sys

def process_file(input_file, output_file):
    """Process a single file and apply the rules"""
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        lines = infile.readlines()
        
        previous_line_empty = False
        for i, line in enumerate(lines):
            # Strip the line to check if it's empty
            stripped_line = line.strip()
            
            # Split columns by tab
            columns = stripped_line.split('\t')
            
            # Rule 1: Delete the entire line if there's only one column with a value of 'O'
            if len(columns) == 1 and columns[0] == "O":
                continue
            
            # Rule 2: Compact multiple empty lines into a single empty line
            if stripped_line == '':
                if previous_line_empty:
                    continue  # Skip multiple empty lines
                previous_line_empty = True
                outfile.write('\n')
            else:
                previous_line_empty = False
            
            # Rule 3: If there are 3 columns, the second column is a single character, and the third column has a value of "O", delete the entire line
            if len(columns) == 3 and len(columns[1]) == 1 and columns[2] == "O":
                continue
            
            # If the line passes all the checks, write it to the output file
            outfile.write(line)

def process_directory(input_dir, output_dir):
    """Recursively process files in the input directory and save to the output directory"""
    for root, dirs, files in os.walk(input_dir):
        # Construct the corresponding output directory
        relative_path = os.path.relpath(root, input_dir)
        output_subdir = os.path.join(output_dir, relative_path)
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
        
        for file in files:
            input_file = os.path.join(root, file)
            output_file = os.path.join(output_subdir, file)
            
            # Process the file
            process_file(input_file, output_file)
            print(f"Processed: {input_file} -> {output_file}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # Check if input_dir exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    # Process the directory
    process_directory(input_dir, output_dir)

if __name__ == "__main__":
    main()
