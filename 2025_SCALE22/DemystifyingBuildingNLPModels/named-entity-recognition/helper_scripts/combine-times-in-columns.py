import os
import re
import argparse

# Define the regex pattern for matching time periods like "am", "pm", etc.
time_periods = re.compile(
    r"^(hours|hrs|hr|days|minutes|mins|min|seconds|secs|sec|years|yrs|yr|am|pm|a\.m\.|p\.m\.|a\.m|p\.m|am\.|pm\.)$",
    re.IGNORECASE,
)

is_time = re.compile(
    r"^(0?[1-9]|1[0-2])[:.][0-5][0-9]$",  # 12-hour format with either : or .
    re.IGNORECASE,
)


def process_file(file_path, output_file_path):
    """
    Process a single file, combining time and period columns when necessary.
    The result is saved in the output directory while preserving the directory structure.
    """
    with open(file_path, "r", encoding="utf-8") as infile, open(
        output_file_path, "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            # Preserve empty lines
            if line.strip() == "":
                outfile.write(line)
                continue

            # Split the line by tabs
            columns = line.strip().split("\t")
            if (
                len(columns) == 3
                and (columns[0] == "0")
                and time_periods.match(columns[1])
            ):
                outfile.write(f"{columns[1]}\t{columns[2]}\n")
            elif ( # If the line has 3 columns, check if the first is numeric and second is a time period
                len(columns) == 3
                and (columns[0].isdigit() or is_time.match(columns[0]))
                and time_periods.match(columns[1])
            ):
                # # Combine the first two columns (time + period) into one
                # combined_time = f"{columns[0]} {columns[1]}"
                # # Keep the third column (TYPE) as is, and write the modified line
                # outfile.write(f"{combined_time}\tB-TIME\n")

                # outfile.write(f"{columns[0]}\tB-TIME\n")
                # outfile.write(f"{columns[1]}\tI-TIME\n")

                # # Split the time into hours and minutes
                if is_time.match(columns[0]):
                    hours, minutes = re.split(r'[:\.]', columns[0])
                    outfile.write(f"{hours}\tB-TIME\n")
                    separator_pos = len(hours)
                    outfile.write(f"{columns[0][separator_pos]}\tI-TIME\n")  # Write the separator (: or .)
                    outfile.write(f"{minutes}\tI-TIME\n")
                    outfile.write(f"{columns[1]}\tI-TIME\n")
                else:
                    outfile.write(f"{columns[0]}\tB-TIME\n")
                    outfile.write(f"{columns[1]}\tI-TIME\n")
            else:
                # Write the line as is if no change is necessary
                outfile.write(line)


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
    parser = argparse.ArgumentParser(description="Process files to combine time and period columns.")
    parser.add_argument("input_dir", help="The input directory containing files to process.")
    parser.add_argument("output_dir", help="The output directory to save processed files.")
    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Run the directory processing with command line arguments
    process_directory(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
