import re
import os
import argparse

def process_times(tokens_tags):
    i = 0
    result = []
    while i < len(tokens_tags):
        token_i, tag_i = tokens_tags[i]
        # Pattern 1: number '.' or ':' number 'am/pm'
        if re.match(r'^\d+$', token_i):
            if i + 3 < len(tokens_tags):
                token_i1, tag_i1 = tokens_tags[i + 1]
                token_i2, tag_i2 = tokens_tags[i + 2]
                token_i3, tag_i3 = tokens_tags[i + 3]
                if token_i1 in ['.', ':'] and re.match(r'^\d+$', token_i2) and re.match(r'^(am|pm|a\.m\.|p\.m\.)$', token_i3, re.IGNORECASE):
                    combined_time = token_i + token_i1 + token_i2
                    result.append((combined_time, tag_i))
                    result.append((token_i3, tag_i3))
                    i += 4
                    continue
        # Pattern 2: number with '.' or ':' and one digit, followed by number and 'am/pm'
        if re.match(r'^\d+[.:]\d$', token_i):
            if i + 2 < len(tokens_tags):
                token_i1, tag_i1 = tokens_tags[i + 1]
                token_i2, tag_i2 = tokens_tags[i + 2]
                if re.match(r'^\d+$', token_i1) and re.match(r'^(am|pm|a\.m\.|p\.m\.)$', token_i2, re.IGNORECASE):
                    combined_time = token_i + token_i1
                    result.append((combined_time, tag_i))
                    result.append((token_i2, tag_i2))
                    i += 3
                    continue
        # Pattern 3: number with '.' or ':', followed by 'number am/pm'
        if re.match(r'^\d+[.:]\d$', token_i):
            if i + 1 < len(tokens_tags):
                token_i1, tag_i1 = tokens_tags[i + 1]
                m = re.match(r'^(\d+)\s*(am|pm|a\.m\.|p\.m\.)$', token_i1, re.IGNORECASE)
                if m:
                    number_part = m.group(1)
                    am_pm_part = m.group(2)
                    combined_time = token_i + number_part
                    result.append((combined_time, tag_i))
                    result.append((am_pm_part, tag_i1))
                    i += 2
                    continue
        # Pattern 4: number followed by 'am/pm'
        if re.match(r'^\d+$', token_i):
            if i + 1 < len(tokens_tags):
                token_i1, tag_i1 = tokens_tags[i + 1]
                if re.match(r'^(am|pm|a\.m\.|p\.m\.)$', token_i1, re.IGNORECASE):
                    result.append((token_i, tag_i))
                    result.append((token_i1, tag_i1))
                    i += 2
                    continue
        # Pattern 5: number with 'am/pm' in the same token
        m = re.match(r'^(\d+)\s*(am|pm|a\.m\.|p\.m\.)$', token_i, re.IGNORECASE)
        if m:
            result.append((token_i, tag_i))
            i += 1
            continue
        # Pattern 6: number with '.' or ':', e.g., '11.41', '9:32'
        if re.match(r'^\d+[.:]\d+$', token_i):
            result.append((token_i, tag_i))
            i += 1
            continue
        # Else, just append the token as is
        result.append((token_i, tag_i))
        i += 1
    return result

def process_file(input_file, output_file):
    sentences = []
    sentence = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line == '':
                if sentence:
                    sentences.append(sentence)
                    sentence = []
                else:
                    # Multiple empty lines, write an empty line to output
                    with open(output_file, 'a', encoding='utf-8') as f_out:
                        f_out.write('\n')
            else:
                # Assuming the token and tag are separated by tabs
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    token = parts[0]
                    tag = '\t'.join(parts[1:])  # In case the tag has tabs
                    sentence.append((token, tag))
                else:
                    # Line doesn't have both token and tag, handle accordingly
                    token = parts[0]
                    tag = ''
                    sentence.append((token, tag))
        # Append the last sentence if any
        if sentence:
            sentences.append(sentence)
    # Now process each sentence
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for sentence in sentences:
            processed_sentence = process_times(sentence)
            for token, tag in processed_sentence:
                f_out.write(f'{token}\t{tag}\n')
            f_out.write('\n')


def process_directory(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        for file in files:
            input_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, input_dir)
            output_file_dir = os.path.join(output_dir, relative_path)
            os.makedirs(output_file_dir, exist_ok=True)
            output_file_path = os.path.join(output_file_dir, file)
            process_file(input_file_path, output_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CoNLL-2003 data to reconstruct times.")
    parser.add_argument("input_dir", help="The input directory containing CoNLL-2003 formatted files.")
    parser.add_argument("output_dir", help="The output directory to save processed files.")
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir)
