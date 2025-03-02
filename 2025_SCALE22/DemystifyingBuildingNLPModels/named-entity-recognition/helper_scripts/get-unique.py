import os


def detect_delimiter(line):
    """Detects whether a line is space or tab-delimited."""
    if "\t" in line:
        return "\t"
    elif " " in line:
        return " "
    else:
        return None  # Handles cases where neither delimiter exists


def collect_unique_values_from_column(input_dir, column_index=1):
    unique_values = set()

    # Loop through all subdirectories and files in the input directory
    for root, _, files in os.walk(input_dir):
        for filename in files:
            file_path = os.path.join(root, filename)

            # Open each file and process it
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    process_file(f, file_path, column_index, unique_values)
            except UnicodeDecodeError:
                print(
                    f"Warning: UTF-8 decoding failed for {file_path}, trying ISO-8859-1."
                )
                try:
                    with open(file_path, "r", encoding="ISO-8859-1") as f:
                        process_file(f, file_path, column_index, unique_values)
                except UnicodeDecodeError:
                    print(
                        f"Error: Could not decode {file_path} with UTF-8 or ISO-8859-1."
                    )
                    continue  # Skip files that can't be decoded

    return unique_values


def process_file(f, file_path, column_index, unique_values):
    delimiter = None
    for line in f:
        # Skip empty lines
        if line.strip() == "":
            continue

        # Detect the delimiter from the first non-empty line
        if delimiter is None:
            delimiter = detect_delimiter(line)
            if delimiter is None:
                print(f"Could not detect delimiter for file {file_path}")
                break

        # Split the line based on the detected delimiter
        columns = line.strip().split(delimiter)

        # Check if there are enough columns
        if len(columns) > column_index:
            # Collect the value(s) from the specified column (2nd column by default)
            tag_list = columns[column_index].split(
                "|"
            )  # Handle the list of values delimited by '|'
            for tag in tag_list:
                unique_values.add(tag)


# Usage example:
input_dir = "NER"  # Replace with the path to your directory
unique_values = collect_unique_values_from_column(input_dir)

# Display the unique values
for value in sorted(unique_values):
    print(value)
