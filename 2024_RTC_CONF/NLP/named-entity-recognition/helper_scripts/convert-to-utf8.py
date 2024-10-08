import os
import argparse

def convert_to_utf8(input_file, output_file, source_encoding):
    """Convert a file from a specific encoding to UTF-8."""
    try:
        with open(input_file, 'r', encoding=source_encoding, errors='replace') as f_in:
            content = f_in.read()

        # Write the content as UTF-8 in the output file
        with open(output_file, 'w', encoding='utf-8') as f_out:
            f_out.write(content)

        print(f"Successfully converted {input_file} from {source_encoding} to UTF-8")
    except Exception as e:
        print(f"Error converting {input_file}: {e}")

def process_all_files(input_dir, output_dir):
    """Recursively process all files in the input directory and convert them to UTF-8."""
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Walk through the input directory
    for root, _, files in os.walk(input_dir):
        for filename in files:
            input_file = os.path.join(root, filename)
            
            # Re-create the directory structure in the output directory
            relative_path = os.path.relpath(root, input_dir)
            target_dir = os.path.join(output_dir, relative_path)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            # Define the output file path
            output_file = os.path.join(target_dir, filename)

            # Try converting from 'ISO-8859-1' or 'cp1252' to UTF-8
            for encoding in ['ISO-8859-1', 'cp1252']:
                try:
                    # Attempt to read the file with the given encoding
                    with open(input_file, 'r', encoding=encoding):
                        # If successful, convert the file to UTF-8
                        convert_to_utf8(input_file, output_file, encoding)
                        break  # Exit after successful conversion
                except UnicodeDecodeError:
                    # If the file can't be decoded with this encoding, try the next
                    continue
                except Exception as e:
                    print(f"Error processing {input_file}: {e}")
                    break  # Stop further processing of this file if a different error occurs

if __name__ == "__main__":

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert files in a directory to UTF-8 encoding.")
    parser.add_argument("input_dir", help="Path to the input directory containing files to be converted.")
    parser.add_argument("output_dir", help="Path to the output directory where converted files will be saved.")

    # Parse arguments
    args = parser.parse_args()

    # Process all files in the input directory and convert them to UTF-8
    process_all_files(args.input_dir, args.output_dir)
