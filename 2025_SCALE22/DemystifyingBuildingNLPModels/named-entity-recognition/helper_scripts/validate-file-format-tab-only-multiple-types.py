import os

# Define the allowed values for the second column
allowed_values = set(
    [
        "B-ANIM",
        "B-Anatomical_system",
        "B-Artist",
        "B-BIO",
        "B-CEL",
        "B-Cell",
        "B-Cellular_component",
        "B-DIS",
        "B-Date",
        "B-Developing_anatomical_structure",
        "B-DocumentReference",
        "B-EVE",
        "B-FOOD",
        "B-Facility",
        "B-INST",
        "B-Immaterial_anatomical_entity",
        "B-LOC",
        "B-Location",
        "B-MEDIA",
        "B-MISC",
        "B-MYTH",
        "B-MilitaryPlatform",
        "B-Misc",
        "B-Money",
        "B-Multi-tissue_structure",
        "B-NORP",
        "B-Nationality",
        "B-ORG",
        "B-Organ",
        "B-Organisation",
        "B-Organism_subdivision",
        "B-Organism_substance",
        "B-Organization",
        "B-PER",
        "B-PLANT",
        "B-Pathological_formation",
        "B-Person",
        "B-Product",
        "B-Quantity",
        "B-TIME",
        "B-Temporal",
        "B-Tissue",
        "B-VEHI",
        "B-Weapon",
        "B-WoA",
        "B-abstract",
        "B-animal",
        "B-company",
        "B-corporation",
        "B-creative-work",
        "B-event",
        "B-facility",
        "B-geo-loc",
        "B-group",
        "B-location",
        "B-movie",
        "B-musicartist",
        "B-object",
        "B-organization",
        "B-other",
        "B-person",
        "B-place",
        "B-plant",
        "B-product",
        "B-quantity",
        "B-sportsteam",
        "B-substance",
        "B-time",
        "B-tvshow",
        "I-ANIM",
        "I-Anatomical_system",
        "I-Artist",
        "I-BIO",
        "I-CEL",
        "I-Cell",
        "I-Cellular_component",
        "I-DIS",
        "I-Date",
        "I-Developing_anatomical_structure",
        "I-DocumentReference",
        "I-EVE",
        "I-FOOD",
        "I-Facility",
        "I-INST",
        "I-Immaterial_anatomical_entity",
        "I-LOC",
        "I-Location",
        "I-MEDIA",
        "I-MISC",
        "I-MYTH",
        "I-MilitaryPlatform",
        "I-Misc",
        "I-Money",
        "I-Multi-tissue_structure",
        "I-NORP",
        "I-Nationality",
        "I-ORG",
        "I-Organ",
        "I-Organisation",
        "I-Organism_subdivision",
        "I-Organism_substance",
        "I-Organization",
        "I-PER",
        "I-PLANT",
        "I-Pathological_formation",
        "I-Person",
        "I-Product",
        "I-Quantity",
        "I-TIME",
        "I-Temporal",
        "I-Tissue",
        "I-VEHI",
        "I-Weapon",
        "I-WoA",
        "I-abstract",
        "I-animal",
        "I-company",
        "I-corporation",
        "I-creative-work",
        "I-event",
        "I-facility",
        "I-geo-loc",
        "I-group",
        "I-location",
        "I-movie",
        "I-musicartist",
        "I-object",
        "I-organization",
        "I-other",
        "I-person",
        "I-place",
        "I-plant",
        "I-product",
        "I-quantity",
        "I-sportsteam",
        "I-substance",
        "I-time",
        "I-tvshow",
        "O",
        "geo-nam",
        "per-giv",
        "org-nam",
        "gpe-nam",
        "per-nam",
        "tim-dat",
        "tim-dow",
        "per-tit",
        "tim-moy",
        "tim-clo",
        "per-fam",
        "tim-yoc",
        "eve-nam",
        "art-nam",
        "org-leg",
        "nat-nam",
        "eve-ord",
        "tim-nam",
        "per-ord",
        "per-ini",
        "tim-dom",
        "per-mid",
        "art-add",
        "B-Rating",
        "B-Actor",
        "B-Plot",
        "B-Organization",
        "I-Rating",
        "B-ACTOR",
        "I-Plot",
        "B-Amenity",
        "I-ACTOR",
        "B-Cuisine",
        "I-Amenity",
        "I-Actor",
        "B-GENRE",
        "B-Genre",
        "B-Restaurant_Name",
        "B-Hours",
        "B-YEAR",
        "I-GENRE",
        "B-Opinion",
        "I-Opinion",
        "I-Restaurant_Name",
        "B-TITLE",
        "I-Hours",
        "B-Award",
        "I-YEAR",
        "I-Genre",
        "B-Price",
        "B-DIRECTOR",
        "I-Award",
        "B-PLOT",
        "I-PLOT",
        "B-Relationship",
        "B-Dish",
        "I-Cuisine",
        "I-DIRECTOR",
        "B-Year",
        "B-RATINGS_AVERAGE",
        "B-Director",
        "I-Dish",
        "I-Price",
        "B-SONG",
        "I-SONG",
        "B-Origin",
        "I-RATINGS_AVERAGE",
        "I-Director",
        "B-REVIEW",
        "I-Origin",
        "I-TITLE",
        "B-CHARACTER",
        "I-Year",
        "B-CHARACTER",
        "B-Quote",
        "I-CHARACTER",
        "B-Soundtrack",
        "B-RATING",
        "I-Quote",
        "B-Character_Name",
        "I-RATING",
        "I-Soundtrack",
        "B-TRAILER",
        "I-Relationship",
        "I-REVIEW",
        "I-Character_Name",
        "I-TRAILER",
        "B-TEMPORAL",
        "I-TEMPORAL",
        "B-AFFILIATION",
        "B-ANATOMICAL",
        "B-ATTRIBUTE",
        "B-BRANDS",
        "B-DATE",
        "B-DOCUMENT",
        "B-DRUG",
        "B-DURATION",
        "B-EVENT",
        "B-FAMILY_NAME",
        "B-GIVEN_NAME",
        "B-LOCATION",
        "B-MEDICAL-CONDITION",
        "B-MONEY",
        "B-NAME",
        "B-NUMERIC",
        "B-ORGANIZATION",
        "B-OTHER",
        "B-PRICE",
        "B-STATUS",
        "B-TIME",
        "I-AFFILIATION",
        "I-ANATOMICAL",
        "I-ATTRIBUTE",
        "I-BRANDS",
        "I-DATE",
        "I-DOCUMENT",
        "I-DRUG",
        "I-DURATION",
        "I-EVENT",
        "I-FAMILY_NAME",
        "I-GIVEN_NAME",
        "I-LOCATION",
        "I-MEDICAL-CONDITION",
        "I-MISC",
        "I-MONEY",
        "I-NAME",
        "I-NUMERIC",
        "I-ORGANIZATION",
        "I-OTHER",
        "I-PRICE",
        "I-STATUS",
        "I-TIME",
        "O",
    ]
)


def detect_delimiter(line):
    """Detects whether a line is tab-delimited."""
    if "\t" in line:
        return "\t"
    # Commenting out the space delimiter detection since we only use tabs now
    # elif ' ' in line:
    #    return ' '
    else:
        return None  # Handles cases where no valid delimiter exists


def validate_file_format(input_dir):
    invalid_files = []

    # Loop through all subdirectories and files in the input directory
    for root, _, files in os.walk(input_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            invalid_lines = []  # To store invalid lines
            line_number = 0  # To track the current line number

            # Open each file and validate it
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    delimiter = None
                    for line in f:
                        line_number += 1

                        # Skip empty lines (sentence boundary)
                        if line.strip() == "":
                            continue

                        # Detect delimiter on the first non-empty line
                        if delimiter is None:
                            delimiter = detect_delimiter(line)
                            if delimiter is None:
                                invalid_lines.append(
                                    f"Delimiter not detected (only tabs allowed) on line {line_number}: {line.strip()}"
                                )
                                continue

                        # Split the line by the detected delimiter
                        columns = line.strip().split(delimiter)

                        # Validate that there are exactly 2 columns
                        if len(columns) != 2:
                            invalid_lines.append(
                                f"Expected 2 columns, but found {len(columns)} on line {line_number}: {line.strip()}"
                            )
                            continue

                        # Validate the second column against allowed values
                        types = columns[1].strip().split("|")
                        for t in types:
                            if t not in allowed_values:
                                invalid_lines.append(
                                    f"Invalid value in second column: {columns[1]} on line {line_number}: {line.strip()}"
                                )
                                continue
            except UnicodeDecodeError:
                invalid_lines.append(
                    f"Failed to decode file using UTF-8 or ISO-8859-1."
                )

            # If there are invalid lines, append the file path and the errors to invalid_files
            if invalid_lines:
                invalid_files.append((file_path, invalid_lines))

    return invalid_files


# Usage example:
input_dir = "NER"  # Replace with the path to your directory
invalid_files = validate_file_format(input_dir)

# Print out the invalid files with reasons and line numbers
if invalid_files:
    print("Files with invalid format:")
    for file, errors in invalid_files:
        print(f"{file}:")
        for error in errors:
            print(f"  {error}")
else:
    print("All files are valid!")
