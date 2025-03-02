import os
import re
import shutil
import sys


def clean_line_format(line):
    """
    Clean the format of the line by:
    1. Converting multiple sequential spaces into a single tab.
    2. Consolidating multiple sequential tabs into a single tab.
    """
    # Convert multiple spaces to a single tab
    line = re.sub(r" +", "\t", line)

    # Convert multiple tabs to a single tab
    line = re.sub(r"\t+", "\t", line)

    return line


def detect_delimiter(line):
    """Detects whether a line is space or tab-delimited."""
    if "\t" in line:
        return "\t"
    elif " " in line:
        return " "
    else:
        return None  # Handles cases where neither delimiter exists


def filter_and_clean_lines(input_dir, output_dir):
    # Copy the directory structure to the output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # Remove the directory if it already exists
    shutil.copytree(
        input_dir, output_dir, ignore=lambda _, __: []
    )  # Copy directory structure only

    # Walk through the input directory to process each file
    for root, _, files in os.walk(input_dir):
        for filename in files:
            input_file_path = os.path.join(root, filename)
            relative_path = os.path.relpath(input_file_path, input_dir)
            output_file_path = os.path.join(output_dir, relative_path)

            # Open input file for reading and output file for writing
            with open(input_file_path, "r", encoding="utf-8") as infile, open(
                output_file_path, "w", encoding="utf-8"
            ) as outfile:
                for line in infile:
                    stripped_line = (
                        line.rstrip()
                    )  # Use rstrip() to remove only trailing whitespace (not leading)

                    # Preserve empty lines
                    if stripped_line == "":
                        outfile.write("\n")  # Write empty line as-is
                        continue

                    # Clean the line format
                    cleaned_line = clean_line_format(stripped_line)

                    # Detect the delimiter after cleaning
                    delimiter = detect_delimiter(cleaned_line)

                    if delimiter is not None:
                        columns = cleaned_line.split(delimiter)

                        # Write lines that contain more than one column
                        if len(columns) > 1:
                            outfile.write(cleaned_line + "\n")


if len(sys.argv) != 3:
    print("Usage: python convert-spaces-to-tabs.py <input_dir> <output_dir>")
    sys.exit(1)

input_dir = sys.argv[1]
output_dir = sys.argv[2]

filter_and_clean_lines(input_dir, output_dir)

print(f"Processed files are saved to {output_dir}.")
