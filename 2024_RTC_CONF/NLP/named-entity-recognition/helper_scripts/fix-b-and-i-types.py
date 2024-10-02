import os
import shutil
import argparse

# List of problematic labels (case insensitive)
labels = [
    "art-add",
    "art-nam",
    "eve-nam",
    "eve-ord",
    "geo-nam",
    "gpe-nam",
    "nat-nam",
    "org-leg",
    "org-nam",
    "per-fam",
    "per-giv",
    "per-ini",
    "per-mid",
    "per-nam",
    "per-ord",
    "per-tit",
    "tim-clo",
    "tim-dat",
    "tim-dom",
    "tim-dow",
    "tim-moy",
    "tim-nam",
    "tim-yoc",
]


# Function to handle the B- and I- labeling
def apply_bio_tags(lines, labels):
    current_label = None
    result = []

    for line in lines:
        if line.strip():  # Non-empty lines
            word, tags = line.strip().split("\t")
            updated_tags = []
            for tag in tags.split("|"):
                tag_lower = tag.lower()
                if tag_lower in labels:
                    if tag_lower == current_label:
                        updated_tags.append(f"I-{tag}")
                    else:
                        updated_tags.append(f"B-{tag}")
                        current_label = tag_lower
                else:
                    updated_tags.append(tag)
                    current_label = None  # Reset current label if it's not in the list

            result.append(f"{word}\t{'|'.join(updated_tags)}\n")
        else:
            result.append("\n")  # Preserve empty lines

    return result


# Function to recursively process files
def process_directory(input_dir, output_dir, labels):
    for root, dirs, files in os.walk(input_dir):
        # Constructing the corresponding output directory path
        relative_path = os.path.relpath(root, input_dir)
        output_subdir = os.path.join(output_dir, relative_path)

        # Create the output subdirectory if it doesn't exist
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        for file in files:
            input_file_path = os.path.join(root, file)
            output_file_path = os.path.join(output_subdir, file)

            with open(input_file_path, "r", encoding="utf-8") as infile:
                lines = infile.readlines()
                processed_lines = apply_bio_tags(lines, labels)

            with open(output_file_path, "w", encoding="utf-8") as outfile:
                outfile.writelines(processed_lines)


# Main execution point
def main():
    parser = argparse.ArgumentParser(description="Process files to fix B- and I- types.")
    parser.add_argument("input_dir", type=str, help="The input directory containing files to process.")
    parser.add_argument("output_dir", type=str, help="The output directory to save processed files.")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    # If output directory exists, clear it; otherwise, create it
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    process_directory(input_dir, output_dir, labels)
    print(f"Processing complete. Files saved in {output_dir}.")


if __name__ == "__main__":
    main()
