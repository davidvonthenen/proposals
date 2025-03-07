import os

def process_file(input_file, output_file):
    encodings = ['utf-8', 'ISO-8859-1', 'cp1252']  # List of encodings to try
    file_processed = False

    for encoding in encodings:
        try:
            # Open the file in binary mode and manually decode lines
            with open(input_file, 'rb') as f_in:
                process_lines(f_in, output_file, encoding)
                file_processed = True
                break  # Stop trying encodings if the file is successfully read
        except Exception as e:
            print(f"Warning: Failed to process {input_file} with {encoding}. Error: {e}")

    if not file_processed:
        print(f"Error: Could not process {input_file} with any known encoding. Skipping file.")

def process_lines(f_in, output_file, encoding):
    # Ensure that the parent directories exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            try:
                # Decode the line using the current encoding
                decoded_line = line.decode(encoding, errors='replace')  # Replace invalid chars

                # Skip empty lines (sentence boundary)
                if decoded_line.strip() == '':
                    f_out.write("\n")
                    continue

                # Split the line into columns (assuming tab-delimited)
                columns = decoded_line.strip().split('\t')

                # Check if there are 3 or more columns
                if len(columns) >= 3:
                    # Write two new lines, one for the 1st and 2nd column, and another for 1st and 3rd column
                    f_out.write(f"{columns[0]}\t{columns[1]}\n")
                    f_out.write(f"{columns[0]}\t{columns[2]}\n")
                else:
                    # Write the line as-is if there are fewer than 3 columns
                    f_out.write(decoded_line)
            except Exception as e:
                print(f"Warning: Could not process line in {input_file}: {e}")

    print(f"Processed file saved to: {output_file}")

def process_all_files(input_dir, output_dir):
    # Loop through all subdirectories and files in the input directory
    for root, _, files in os.walk(input_dir):
        for filename in files:
            input_file = os.path.join(root, filename)

            # Compute the relative path to preserve directory structure
            relative_path = os.path.relpath(input_file, input_dir)
            output_file = os.path.join(output_dir, relative_path)

            # Process each file
            process_file(input_file, output_file)

# Usage example:
input_dir = '/path/to/your/input_directory'  # Replace with the path to your input directory
output_dir = '/path/to/your/output_directory'  # Replace with the path to your output directory

process_all_files(input_dir, output_dir)
