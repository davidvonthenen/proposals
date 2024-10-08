import os
import re
import argparse

# Regular expressions for date components
months_regex = re.compile(
    r"^(January|Jan|February|Feb|March|Mar|April|Apr|May|June|Jun|July|Jul|August|Aug|September|Sept|October|Oct|November|Nov|December|Dec)$",
    re.IGNORECASE,
)

day_regex = re.compile(r"^([1-9]|[12][0-9]|3[01])(st|nd|rd|th)?$", re.IGNORECASE)

year_regex = re.compile(r"^[1-2][0-9]{3}$")

of_regex = re.compile(r"^of$", re.IGNORECASE)

comma_regex = re.compile(r"^,$")

# Existing regex patterns
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


def check_pattern_month_day_year(tokens):
    if len(tokens) < 3:
        return False
    t1, t2, t3 = tokens[0], tokens[1], tokens[2]
    token1 = t1[1][0]
    token2 = t2[1][0]
    token3 = t3[1][0]
    return (
        months_regex.match(token1)
        and day_regex.match(token2)
        and year_regex.match(token3)
    )


def check_pattern_month_day_comma_year(tokens):
    if len(tokens) < 4:
        return False
    t1, t2, t3, t4 = tokens[0], tokens[1], tokens[2], tokens[3]
    token1 = t1[1][0]
    token2 = t2[1][0]
    token3 = t3[1][0]
    token4 = t4[1][0]
    return (
        months_regex.match(token1)
        and day_regex.match(token2)
        and comma_regex.match(token3)
        and year_regex.match(token4)
    )


def check_pattern_day_of_month_year(tokens):
    if len(tokens) < 4:
        return False
    t1, t2, t3, t4 = tokens[0], tokens[1], tokens[2], tokens[3]
    token1 = t1[1][0]
    token2 = t2[1][0]
    token3 = t3[1][0]
    token4 = t4[1][0]
    return (
        day_regex.match(token1)
        and of_regex.match(token2)
        and months_regex.match(token3)
        and year_regex.match(token4)
    )


def check_pattern_day_of_month_comma_year(tokens):
    if len(tokens) < 5:
        return False
    t1, t2, t3, t4, t5 = tokens[0], tokens[1], tokens[2], tokens[3], tokens[4]
    token1 = t1[1][0]
    token2 = t2[1][0]
    token3 = t3[1][0]
    token4 = t4[1][0]
    token5 = t5[1][0]
    return (
        day_regex.match(token1)
        and of_regex.match(token2)
        and months_regex.match(token3)
        and comma_regex.match(token4)
        and year_regex.match(token5)
    )


def output_date_tokens(outfile, tokens):
    for i, (line, columns) in enumerate(tokens):
        token = columns[0]
        label = "B-DATE" if i == 0 else "I-DATE"
        outfile.write(f"{token}\t{label}\n")


def process_token(outfile, token_info):
    line, columns = token_info
    token = columns[0] if columns else ""
    # Check for number-duration on the same line
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

    # Process any remaining previous_token_info
    if previous_token_info:
        process_token(outfile, previous_token_info)


def process_file(file_path, output_file_path):
    with open(file_path, "r", encoding="utf-8") as infile, open(
        output_file_path, "w", encoding="utf-8"
    ) as outfile:
        buffer = []
        for line in infile:
            if line.strip() == "":
                # Flush the buffer
                process_buffer(outfile, buffer)
                buffer = []
                # Write the empty line
                outfile.write(line)
                continue

            columns = line.strip().split("\t")
            token = columns[0] if columns else ""
            token_info = (line, columns)
            buffer.append(token_info)

            # Attempt to match date patterns starting from the beginning of the buffer
            matched = False
            for i in range(5, 2, -1):  # Check from 5 tokens down to 3
                if len(buffer) >= i:
                    if i == 5 and check_pattern_day_of_month_comma_year(buffer[:5]):
                        output_date_tokens(outfile, buffer[:5])
                        buffer = buffer[5:]
                        matched = True
                        break
                    elif i == 4:
                        if check_pattern_day_of_month_year(buffer[:4]):
                            output_date_tokens(outfile, buffer[:4])
                            buffer = buffer[4:]
                            matched = True
                            break
                        elif check_pattern_month_day_comma_year(buffer[:4]):
                            output_date_tokens(outfile, buffer[:4])
                            buffer = buffer[4:]
                            matched = True
                            break
                    elif i == 3 and check_pattern_month_day_year(buffer[:3]):
                        output_date_tokens(outfile, buffer[:3])
                        buffer = buffer[3:]
                        matched = True
                        break

            if matched:
                continue

            # If buffer length exceeds 5, process the first token
            if len(buffer) > 5:
                process_token(outfile, buffer.pop(0))

        # After the loop, process any remaining tokens
        process_buffer(outfile, buffer)


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
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Process files to combine times and dates in lines."
    )
    parser.add_argument(
        "input_dir", help="The input directory containing files to process."
    )
    parser.add_argument(
        "output_dir", help="The output directory to save processed files."
    )

    # Parse arguments
    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Run the directory processing with provided arguments
    process_directory(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
