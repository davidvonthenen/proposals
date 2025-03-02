import os
import re
import sys


def modify_file(input_file, output_file, search_value, new_value):
    """Modify the second column based on the case-insensitive match of the first column."""
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            if line.strip():  # If the line contains content
                # Check if the delimiter is a space and convert to tab
                if re.search(
                    r" \S+", line
                ):  # Checks for a space followed by non-space characters
                    line = re.sub(
                        r" +", "\t", line, count=1
                    )  # Replace spaces with a single tab

                columns = line.split("\t")
                if len(columns) == 2:
                    word, label = columns[0], columns[1].strip()
                    if word.lower() == search_value.lower():
                        outfile.write(f"{word}\t{new_value}\n")
                    else:
                        outfile.write(line)  # Write the line unchanged if no match
                else:
                    outfile.write(line)  # In case of unexpected formatting
            else:
                outfile.write(line)  # Preserve empty lines


def process_directory(input_dir, output_dir, search_value, new_value):
    """Recursively process all files in the input directory and write modified files to the output directory."""
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            input_file = os.path.join(root, file)
            # Recreate the same directory structure in the output_dir
            relative_path = os.path.relpath(input_file, input_dir)
            output_file = os.path.join(output_dir, relative_path)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            modify_file(input_file, output_file, search_value, new_value)


if __name__ == "__main__":
    # Ensure the necessary parameters are passed
    if len(sys.argv) < 3:
        print("Usage: python script_name.py <search_value> <new_value>")
        sys.exit(1)

    # Get the search value and the new value from the command line
    search_value = sys.argv[1]
    new_value = sys.argv[2]

    # Define the input and output directories (can be changed as needed)
    input_directory = (
        "NER"  # Change this to the path where the original files are located
    )
    output_directory = (
        "NER_NEW"  # Change this to the path where you want the modified files saved
    )

    # Process the directory with the provided search value and new value
    process_directory(input_directory, output_directory, search_value, new_value)

    print(f"Processing complete. Modified files are saved in {output_directory}")
