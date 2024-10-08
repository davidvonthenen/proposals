import os
import re
import argparse

# Regular expressions for date components and number-duration patterns
months_regex = re.compile(
    r"^(January|Jan|February|Feb|March|Mar|April|Apr|May|June|Jun|July|Jul|August|Aug|September|Sept|October|Oct|November|Nov|December|Dec)$",
    re.IGNORECASE,
)

day_regex = re.compile(r"^([1-9]|[12][0-9]|3[01])(st|nd|rd|th)?$", re.IGNORECASE)
year_regex = re.compile(r"^[1-2][0-9]{3}$")
of_regex = re.compile(r"^of$", re.IGNORECASE)
comma_regex = re.compile(r"^,$")

number_regex = re.compile(
    r"^(?:\d+|"
    r"zero|one|two|three|four|five|six|seven|eight|nine|"
    r"ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty(?:-one|-two|-three|-four|-five|-six|-seven|-eight|-nine)?|"
    r"thirty(?:-one|-two|-three|-four|-five|-six|-seven|-eight|-nine)?|"
    r"forty(?:-one|-two|-three|-four|-five|-six|-seven|-eight|-nine)?|"
    r"fifty(?:-one|-two|-three|-four|-five|-six|-seven|-eight|-nine)?|"
    r"twenty|thirty|forty|fifty"
    r")$",
    re.IGNORECASE,
)

duration_regex = re.compile(
    r"^(hours?|hrs?|hr?|days?|minutes?|mins?|seconds?|secs?|sec?|years?|yrs?|yr?)$",
    re.IGNORECASE,
)
time_regex = re.compile(r"^(am|pm|a\.m\.|a\.m|p\.m\.|p\.m)$", re.IGNORECASE)

# Regex for combining numbers with durations (e.g., "3 days")
number_duration_regex = re.compile(
    r"^(?P<number>(?:\d+|"
    r"zero|one|two|three|four|five|six|seven|eight|nine|"
    r"ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty(?:-one|-two|-three|-four|-five|-six|-seven|-eight|-nine)?|"
    r"thirty(?:-one|-two|-three|-four|-five|-six|-seven|-eight|-nine)?|"
    r"forty(?:-one|-two|-three|-four|-five|-six|-seven|-eight|-nine)?|"
    r"fifty(?:-one|-two|-three|-four|-five|-six|-seven|-eight|-nine)?|"
    r"twenty|thirty|forty|fifty"
    r"))-(?P<duration>"
    r"years?|yrs?|months?|mos?|weeks?|wks?|days?|hrs?|hours?|minutes?|mins?|seconds?|secs?"
    r")$",
    re.IGNORECASE,
)


def process_token(outfile, token_info):
    line, columns = token_info
    token = columns[0] if columns else ""
    match = number_duration_regex.match(token)
    if match:
        number_part = match.group("number")
        duration_part = match.group("duration")
        # Write the number part with B-DURATION
        outfile.write(f"{number_part}\tB-DURATION\n")
        # Write the duration part with I-DURATION
        outfile.write(f"{duration_part}\tI-DURATION\n")
    else:
        # Write the line as is
        outfile.write(line)


def process_buffer(outfile, buffer):
    previous_token_info = None
    while buffer:
        token_info = buffer.pop(0)
        line, columns = token_info
        token = columns[0] if columns else ""

        if previous_token_info:
            prev_line, prev_columns = previous_token_info
            prev_token = prev_columns[0] if prev_columns else ""

            # Check for number-duration on separate lines
            if number_regex.match(prev_token) and duration_regex.match(token):
                # Output the number with B-DURATION
                outfile.write(f"{prev_token}\tB-DURATION\n")
                # Output the duration with I-DURATION
                outfile.write(f"{token}\tI-DURATION\n")
                previous_token_info = None
                continue
            # Check for number-time
            if number_regex.match(prev_token) and time_regex.match(token):
                outfile.write(f"{prev_token}\tB-TIME\n")
                outfile.write(f"{token}\tI-TIME\n")
                previous_token_info = None
                continue

            # Output previous token
            process_token(outfile, previous_token_info)

        previous_token_info = token_info

    # Process any remaining token
    if previous_token_info:
        process_token(outfile, previous_token_info)


def process_file(file_path, output_file_path):
    with open(file_path, "r", encoding="utf-8") as infile, open(
        output_file_path, "w", encoding="utf-8"
    ) as outfile:
        buffer = []
        for line in infile:
            if line.strip() == "":
                # Sentence boundary; process the buffer
                process_buffer(outfile, buffer)
                buffer = []
                # Write the empty line
                outfile.write(line)
                continue

            columns = line.strip().split("\t")
            token_info = (line, columns)
            buffer.append(token_info)

        # Process any remaining tokens in the buffer
        process_buffer(outfile, buffer)


def process_directory(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        rel_path = os.path.relpath(root, input_dir)
        output_root = os.path.join(output_dir, rel_path)
        if not os.path.exists(output_root):
            os.makedirs(output_root)

        for filename in files:
            input_file_path = os.path.join(root, filename)
            output_file_path = os.path.join(output_root, filename)
            process_file(input_file_path, output_file_path)


def main():
    parser = argparse.ArgumentParser(
        description="Process files to combine times and dates in lines."
    )
    parser.add_argument(
        "input_dir", help="The input directory containing files to process."
    )
    parser.add_argument(
        "output_dir", help="The output directory to save processed files."
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    process_directory(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
