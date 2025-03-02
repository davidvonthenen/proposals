import os
import argparse


# Define the key-value replacements as a dictionary
replacements = {
    # start of PII
    # names
    "NAME": "B-NAME",
    "B-ART-ADD": "B-NAME",
    "I-ART-ADD": "I-NAME",
    "B-ART-NAM": "B-NAME",
    "I-ART-NAM": "I-NAME",
    "B-PERSON": "B-GIVEN_NAME",
    "I-PERSON": "I-FAMILY_NAME",
    "B-PER": "B-GIVEN_NAME",
    "I-PER": "I-FAMILY_NAME",
    "B-PER-GIV": "B-GIVEN_NAME",
    "I-PER-GIV": "I-GIVEN_NAME",
    "B-PER-FAM": "B-FAMILY_NAME",
    "I-PER-FAM": "I-FAMILY_NAME",
    "B-ACTOR": "B-GIVEN_NAME",
    "I-ACTOR": "I-FAMILY_NAME",
    "B-DIRECTOR": "B-GIVEN_NAME",
    "I-DIRECTOR": "I-FAMILY_NAME",
    "B-MUSICARTIST": "B-NAME",
    "I-MUSICARTIST": "I-NAME",
    "B-CHARACTER_NAME": "B-NAME",
    "I-CHARACTER_NAME": "I-NAME",
    "B-ARTIST": "B-NAME",
    "I-ARTIST": "I-NAME",
    "B-CHARACTER": "B-NAME",
    "I-CHARACTER": "I-NAME",
    "B-MYTH": "B-NAME",
    "I-MYTH": "I-NAME",
    "B-PER-NAM": "B-NAME",
    "I-PER-NAM": "I-NAME",
    "B-PER-ORD": "B-ATTRIBUTE",
    "I-PER-ORD": "I-ATTRIBUTE",
    "B-PER-TIT": "B-ATTRIBUTE",
    "I-PER-TIT": "I-ATTRIBUTE",
    # organization
    "B-COMPANY": "B-ORGANIZATION",
    "I-COMPANY": "I-ORGANIZATION",
    "B-CORPORATION": "B-ORGANIZATION",
    "I-CORPORATION": "I-ORGANIZATION",
    "B-ORGANISATION": "B-ORGANIZATION",
    "I-ORGANISATION": "I-ORGANIZATION",
    "B-ORGANIZATION": "B-ORGANIZATION",
    "I-ORGANIZATION": "I-ORGANIZATION",
    "B-ORG": "B-ORGANIZATION",
    "I-ORG": "I-ORGANIZATION",
    "B-GROUP": "B-ORGANIZATION",
    "I-GROUP": "I-ORGANIZATION",
    "B-RESTAURANT_NAME": "B-ORGANIZATION",
    "I-RESTAURANT_NAME": "I-ORGANIZATION",
    "B-SPORTSTEAM": "B-ORGANIZATION",
    "I-SPORTSTEAM": "I-ORGANIZATION",
    "B-ORG-LEG": "B-ORGANIZATION",
    "I-ORG-LEG": "I-ORGANIZATION",
    "B-ORG-NAM": "B-ORGANIZATION",
    "I-ORG-NAM": "I-ORGANIZATION",
    "B-NATIONALITY": "B-AFFILIATION",
    "I-NATIONALITY": "I-AFFILIATION",
    "B-NORP": "B-AFFILIATION",
    "I-NORP": "I-AFFILIATION",
    "B-GPE-NAM": "B-AFFILIATION",
    "I-GPE-NAM": "I-AFFILIATION",
    # location
    "B-LOC": "B-LOCATION",
    "I-LOC": "I-LOCATION",
    "B-LOC": "B-LOCATION",
    "I-LOC": "I-LOCATION",
    "B-PLACE": "B-LOCATION",
    "I-PLACE": "I-LOCATION",
    "B-GEO-LOC": "B-LOCATION",
    "I-GEO-LOC": "I-LOCATION",
    "B-FACILITY": "B-LOCATION",
    "I-FACILITY": "I-LOCATION",
    "B-CEL": "B-LOCATION",
    "I-CEL": "I-LOCATION",
    "B-GEO-NAM": "B-LOCATION",
    "I-GEO-NAM": "I-LOCATION",
    # end of PII
    # start of GEN
    # time
    "B-DATE": "B-DATE",
    "I-DATE": "I-DATE",
    "B-TIM-DOW": "B-DATE",
    "I-TIM-DOW": "I-DATE",
    "B-TIM-MOY": "B-DATE",
    "I-TIM-MOY": "I-DATE",
    "B-TIM-NAM": "B-DATE",
    "I-TIM-NAM": "I-DATE",
    "B-TIM-YOC": "B-DATE",
    "I-TIM-YOC": "I-DATE",
    "B-HOURS": "B-DURATION",
    "I-HOURS": "I-DURATION",
    "B-YEAR": "B-DURATION",
    "I-YEAR": "I-DURATION",
    "B-TEMPORAL": "B-DURATION",
    "I-TEMPORAL": "I-DURATION",
    "B-TIME": "B-TIME",
    "I-TIME": "I-TIME",
    "B-TIM-DAT": "B-TIME",
    "I-TIM-DAT": "I-TIME",
    "I-TIM-CLO": "B-TIME",
    "B-TIM-CLO": "I-TIME",
    "B-EVE-NAM": "B-EVENT",
    "I-EVE-NAM": "I-EVENT",
    "B-EVENT": "B-EVENT",
    "I-EVENT": "I-EVENT",
    "B-EVE": "B-EVENT",
    "I-EVE": "I-EVENT",
    "B-VEHI": "B-NAME",
    "I-VEHI": "I-NAME",
    "B-PLANT": "B-NAME",
    "I-PLANT": "I-NAME",
    "B-AWARD": "B-NAME",
    "I-AWARD": "I-NAME",
    "B-CUISINE": "B-NAME",
    "I-CUISINE": "I-NAME",
    "B-DISH": "B-NAME",
    "I-DISH": "I-NAME",
    "B-FOOD": "B-NAME",
    "I-FOOD": "I-NAME",
    "B-OBJECT": "B-OTHER",
    "I-OBJECT": "I-OTHER",
    "B-NAT-NAM": "B-OTHER",
    "I-NAT-NAM": "I-OTHER",
    "B-QUANTITY": "B-NUMERIC",
    "I-QUANTITY": "I-NUMERIC",
    "B-TIM-DOM": "B-NUMERIC",
    "I-TIM-DOM": "I-NUMERIC",
    "B-MISC": "B-BRANDS",
    "B-MISC": "I-BRANDS",
    "B-INST": "B-BRANDS",
    "I-INST": "I-BRANDS",
    "B-PRODUCT": "B-BRANDS",
    "I-PRODUCT": "I-BRANDS",
    # documents
    "B-DOCUMENTREFERENCE": "B-DOCUMENT",
    "I-DOCUMENTREFERENCE": "I-DOCUMENT",
    # end of GEN
    # start of PCI
    # currency
    "B-MONEY": "B-MONEY",
    "I-MONEY": "I-MONEY",
    "B-PRICE": "B-PRICE",
    "I-PRICE": "I-PRICE",
    # end of PCI
    # start of PHI
    # healthcare
    "B-RATING": "B-STATUS",
    "I-RATING": "I-STATUS",
    "B-RATINGS_AVERAGE": "B-STATUS",
    "I-RATINGS_AVERAGE": "I-STATUS",
    "B-REVIEW": "B-STATUS",
    "I-REVIEW": "I-STATUS",
    "B-OPINION": "B-STATUS",
    "I-OPINION": "I-STATUS",
    "B-ANATOMICAL_SYSTEM": "B-ANATOMICAL",
    "I-ANATOMICAL_SYSTEM": "I-ANATOMICAL",
    "B-CELLULAR_COMPONENT": "B-ANATOMICAL",
    "I-CELLULAR_COMPONENT": "I-ANATOMICAL",
    "B-DEVELOPING_ANATOMICAL_STRUCTURE": "B-ANATOMICAL",
    "I-DEVELOPING_ANATOMICAL_STRUCTURE": "I-ANATOMICAL",
    "B-IMMATERIAL_ANATOMICAL_ENTITY": "B-ANATOMICAL",
    "I-IMMATERIAL_ANATOMICAL_ENTITY": "I-ANATOMICAL",
    "B-MULTI-TISSUE_STRUCTURE": "B-ANATOMICAL",
    "I-MULTI-TISSUE_STRUCTURE": "I-ANATOMICAL",
    "B-ORGAN": "B-ANATOMICAL",
    "I-ORGAN": "I-ANATOMICAL",
    "B-ORGANISM_SUBDIVISION": "B-ANATOMICAL",
    "I-ORGANISM_SUBDIVISION": "I-ANATOMICAL",
    "B-ORGANISM_SUBSTANCE": "B-ANATOMICAL",
    "I-ORGANISM_SUBSTANCE": "I-ANATOMICAL",
    "B-CELL": "B-ANATOMICAL",
    "I-CELL": "I-ANATOMICAL",
    "B-TISSUE": "B-ANATOMICAL",
    "I-TISSUE": "I-ANATOMICAL",
    "B-PATHOLOGICAL_FORMATION": "B-MEDICAL-CONDITION",
    "I-PATHOLOGICAL_FORMATION": "I-MEDICAL-CONDITION",
    "B-DIS": "B-MEDICAL-CONDITION",
    "I-DIS": "I-MEDICAL-CONDITION",
    "B-BIO": "B-MEDICAL-CONDITION",
    "I-BIO": "I-MEDICAL-CONDITION",
    "B-SUBSTANCE": "B-DRUG",
    "I-SUBSTANCE": "I-DRUG",
    # end of PHI
    # other
    "B-MEDIA": "O",
    "I-MEDIA": "O",
    "B-TITLE": "O",
    "I-TITLE": "O",
    "B-SONG": "O",
    "I-SONG": "O",
    "B-SOUNDTRACK": "O",
    "I-SOUNDTRACK": "O",
    "B-TRAILER": "O",
    "I-TRAILER": "O",
    "B-CREATIVE-WORK": "O",
    "I-CREATIVE-WORK": "O",
    "B-MOVIE": "O",
    "I-MOVIE": "O",
    "B-TVSHOW": "O",
    "I-TVSHOW": "O",
    "B-GENRE": "O",
    "I-GENRE": "O",
    "B-PLOT": "O",
    "I-PLOT": "O",
    "B-QUOTE": "O",
    "I-QUOTE": "O",
    "B-ABSTRACT": "O",
    "I-ABSTRACT": "O",
    "B-ORIGIN": "O",
    "I-ORIGIN": "O",
    "B-ANIMAL": "O",
    "I-ANIMAL": "O",
    "B-ANIM": "O",
    "I-ANIM": "O",
    "B-MILITARYPLATFORM": "O",
    "I-MILITARYPLATFORM": "O",
    "B-AMENITY": "O",
    "I-AMENITY": "O",
    "B-OTHER": "O",
    "I-OTHER": "O",
    "B-RELATIONSHIP": "O",
    "I-RELATIONSHIP": "O",
    "B-EVE-ORD": "O",
    "I-EVE-ORD": "O",
    "B-PER-INI": "O",
    "I-PER-INI": "O",
    "B-PER-MID": "O",
    "I-PER-MID": "O",
    "B-WEAPON": "O",
    "I-WEAPON": "O",
    "B-WOA": "O",
    "I-WOA": "O",
    "B-ARTIST_OR_WOA": "O",
    "I-ARTIST_OR_WOA": "O",
}


# Function to replace named entity tags based on the lookup table
def replace_tags(tag, replacements):
    tag_upper = tag.upper()
    for key, value in replacements.items():
        if tag_upper == key.upper():
            return value
    return tag_upper


# Process each file in the directory recursively
def process_directory(input_dir, output_dir, replacements):
    # Walk through the input directory
    for root, dirs, files in os.walk(input_dir):
        # Create corresponding directories in output folder
        relative_path = os.path.relpath(root, input_dir)
        ner_root = os.path.join(output_dir, relative_path)
        os.makedirs(ner_root, exist_ok=True)

        # Process each file in the directory
        for file_name in files:
            input_file_path = os.path.join(root, file_name)
            output_file_path = os.path.join(ner_root, file_name)

            # Read the file and process each line
            with open(input_file_path, "r", encoding="utf-8") as infile, open(
                output_file_path, "w", encoding="utf-8"
            ) as outfile:
                for line in infile:
                    if line.strip() == "":
                        outfile.write("\n")
                        continue

                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        word, tags = parts
                        # Replace each tag in the list using the lookup table
                        replaced_tags = "|".join(
                            [replace_tags(tag, replacements) for tag in tags.split("|")]
                        )
                        outfile.write(f"{word}\t{replaced_tags}\n")
                    else:
                        # Write the line unchanged if it does not match the expected format
                        outfile.write(line)


# Define the input and output directories
def main():
    parser = argparse.ArgumentParser(description="Process NER tags in files.")
    parser.add_argument(
        "input_dir", type=str, help="Input directory containing files to process"
    )
    parser.add_argument(
        "output_dir", type=str, help="Output directory to save processed files"
    )
    args = parser.parse_args()

    # Run the processing function
    process_directory(args.input_dir, args.output_dir, replacements)


if __name__ == "__main__":
    main()
