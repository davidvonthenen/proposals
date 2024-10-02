#!/bin/bash

# set -o errexit
set -o nounset
set -o pipefail
set -o xtrace

# clean up unwanted files
pushd "./NER" || exit 1
    find . -name "*.md" -type f -delete
    find . -name "*DS_Store*" -type f -delete
popd || exit 1

pushd "./NER/CoNLL-2003" || exit 1
    find . -type f | grep -v "openNLP" | xargs rm
popd || exit 1

pushd "./NER/GMB" || exit 1
    find . -name "*.xml" -type f -delete
    find . -name "*.drg" -type f -delete
    find . -name "*.met" -type f -delete
    find . -name "*.raw" -type f -delete
    find . -name "*.iob" -type f -delete
    find . -name "*.off" -type f -delete
popd || exit 1

pushd "./NER/Ritter" || exit 1
    find . -name "*.py" -type f -delete
    find . -name "*.txt" -type f -delete
    find . -name "*notypes*" -type f -delete
popd || exit 1

pushd "./NER/Music-NER" || exit 1
    find . -name "*.csv" -type f -delete
popd || exit 1

pushd "./NER/MultiNERD" || exit 1
    find . -name "*_zh*" -type f -delete
    find . -name "*_nl*" -type f -delete
    find . -name "*_pl*" -type f -delete
    find . -name "*_ru*" -type f -delete
    find . -name "*_de*" -type f -delete
    find . -name "*_es*" -type f -delete
    find . -name "*_pt*" -type f -delete
    find . -name "*_fr*" -type f -delete
    find . -name "*_it*" -type f -delete
popd || exit 1

# flatten directory structure
pushd "./NER" || exit 1
    cp ../helper_scripts/flatten-directory-structure.py ./

    python flatten-directory-structure.py GMB GMB_NEW
    if [[ -d "./GMB_NEW" ]]; then
        rm -rf ./GMB
    fi
    mv ./GMB_NEW ./GMB

    python flatten-directory-structure.py GUM GUM_NEW
    if [[ -d "./GUM_NEW" ]]; then
        rm -rf ./GUM
    fi
    mv ./GUM_NEW ./GUM

    python flatten-directory-structure.py MultiNERD MultiNERD_NEW
    if [[ -d "./MultiNERD_NEW" ]]; then
        rm -rf ./MultiNERD
    fi
    mv ./MultiNERD_NEW ./MultiNERD

    python flatten-directory-structure.py Music-NER Music-NER_NEW
    if [[ -d "./Music-NER_NEW" ]]; then
        rm -rf ./Music-NER
    fi
    mv ./Music-NER_NEW ./Music-NER

    python flatten-directory-structure.py re3d re3d_NEW
    if [[ -d "./re3d_NEW" ]]; then
        rm -rf ./re3d
    fi
    mv ./re3d_NEW ./re3d

    python flatten-directory-structure.py SEC-filings SEC-filings_NEW
    if [[ -d "./SEC-filings" ]]; then
        rm -rf ./SEC-filings
    fi
    mv ./SEC-filings_NEW ./SEC-filings

    python flatten-directory-structure.py WNUT17 WNUT17_NEW
    if [[ -d "./WNUT17_NEW" ]]; then
        rm -rf ./WNUT17
    fi
    mv ./WNUT17_NEW ./WNUT17

    rm ./flatten-directory-structure.py
popd || exit 1

# convert jsonl to conll2003
pushd "./NER" || exit 1
    cp ../helper_scripts/convert-jsonl-to-conll2003.py ./

    python convert-jsonl-to-conll2003.py MultiNERD MultiNERD_NEW
    if [[ -d "./MultiNERD_NEW" ]]; then
        rm -rf ./MultiNERD
    fi
    mv ./MultiNERD_NEW ./MultiNERD

    rm ./convert-jsonl-to-conll2003.py
popd || exit 1

# convert spaces to tabs
cp ./helper_scripts/convert-spaces-to-tabs.py ./

python convert-spaces-to-tabs.py NER NER_NEW
if [[ -d "./NER_NEW" ]]; then
    rm -rf ./NER
fi
mv ./NER_NEW ./NER

rm ./convert-spaces-to-tabs.py

# delete empty lines


# change encoding to utf-8
cp ./helper_scripts/convert-to-utf8.py ./

python convert-to-utf8.py NER NER_NEW
if [[ -d "./NER_NEW" ]]; then
    rm -rf ./NER
fi
mv ./NER_NEW ./NER

rm ./convert-to-utf8.py

# clean up docstart
cp ./helper_scripts/clean-up-docstart.py ./

python clean-up-docstart.py NER NER_NEW
if [[ -d "./NER_NEW" ]]; then
    rm -rf ./NER
fi
mv ./NER_NEW ./NER

rm ./clean-up-docstart.py

# remove url from first line
cp ./helper_scripts/remove-url-from-first-line.py ./

python remove-url-from-first-line.py

rm ./remove-url-from-first-line.py

# convert from 4 to 2 columns
pushd "./NER" || exit 1
    cp ../helper_scripts/convert-from-4-to-2-columns.py ./

    python convert-from-4-to-2-columns.py AnEM AnEM_NEW
    if [[ -d "./AnEM_NEW" ]]; then
        rm -rf ./AnEM
    fi
    mv ./AnEM_NEW ./AnEM

    python convert-from-4-to-2-columns.py CoNLL-2003 CoNLL-2003_NEW
    if [[ -d "./CoNLL-2003_NEW" ]]; then
        rm -rf ./CoNLL-2003
    fi
    mv ./CoNLL-2003_NEW ./CoNLL-2003

    python convert-from-4-to-2-columns.py GMB GMB_NEW
    if [[ -d "./GMB_NEW" ]]; then
        rm -rf ./GMB
    fi
    mv ./GMB_NEW ./GMB

    python convert-from-4-to-2-columns.py MultiNERD MultiNERD_NEW
    if [[ -d "./MultiNERD_NEW" ]]; then
        rm -rf ./MultiNERD
    fi
    mv ./MultiNERD_NEW ./MultiNERD

    python convert-from-4-to-2-columns.py SEC-filings SEC-filings_NEW
    if [[ -d "./SEC-filings_NEW" ]]; then
        rm -rf ./SEC-filings
    fi
    mv ./SEC-filings_NEW ./SEC-filings

    rm ./convert-from-4-to-2-columns.py
popd || exit 1

# flip the reminaing columns
pushd "./NER" || exit 1
    cp ../helper_scripts/flip-word-and-type-columns.py ./

    python flip-word-and-type-columns.py MITMovie MITMovie_NEW
    if [[ -d "./MITMovie_NEW" ]]; then
        rm -rf ./MITMovie
    fi
    mv ./MITMovie_NEW ./MITMovie

    python flip-word-and-type-columns.py MITRestaurant MITRestaurant_NEW
    if [[ -d "./MITRestaurant_NEW" ]]; then
        rm -rf ./MITRestaurant
    fi
    mv ./MITRestaurant_NEW ./MITRestaurant

    rm ./flip-word-and-type-columns.py
popd || exit 1

# associate fix times and days
cp ./helper_scripts/fix-weird-errors.py ./

python fix-weird-errors.py NER NER_NEW
if [[ -d "./NER_NEW" ]]; then
    rm -rf ./NER
fi
mv ./NER_NEW ./NER

rm ./fix-weird-errors.py

# associate fix times and days
cp ./helper_scripts/fix-time-appearing-on-multiple-lines.py ./

python fix-time-appearing-on-multiple-lines.py NER NER_NEW
if [[ -d "./NER_NEW" ]]; then
    rm -rf ./NER
fi
mv ./NER_NEW ./NER

rm ./fix-time-appearing-on-multiple-lines.py

# associate fix times and days
cp ./helper_scripts/combine-times-in-columns.py ./

python combine-times-in-columns.py NER NER_NEW
if [[ -d "./NER_NEW" ]]; then
    rm -rf ./NER
fi
mv ./NER_NEW ./NER

rm ./combine-times-in-columns.py

# associate times and dates
cp ./helper_scripts/associate-times-and-dates.py ./

python associate-times-and-dates.py NER NER_NEW
if [[ -d "./NER_NEW" ]]; then
    rm -rf ./NER
fi
mv ./NER_NEW ./NER

rm ./associate-times-and-dates.py

# consolidate the types to change the second column in multiple values with "|"
cp ./helper_scripts/combine_multiple_ner_types_into_single_column.py ./

# do this once
python combine_multiple_ner_types_into_single_column.py NER NER_NEW
if [[ -d "./NER_NEW" ]]; then
    rm -rf ./NER
fi
mv ./NER_NEW ./NER

# do this twice
python combine_multiple_ner_types_into_single_column.py NER NER_NEW
if [[ -d "./NER_NEW" ]]; then
    rm -rf ./NER
fi
mv ./NER_NEW ./NER

rm ./combine_multiple_ner_types_into_single_column.py

# change the odd NER types to B- and I- types
cp ./helper_scripts/fix-b-and-i-types.py ./

python fix-b-and-i-types.py NER NER_NEW
if [[ -d "./NER_NEW" ]]; then
    rm -rf ./NER
fi
mv ./NER_NEW ./NER

rm ./fix-b-and-i-types.py

# consoldiate the types to standard NER types
cp ./helper_scripts/consolidate-types.py ./

# run once
python consolidate-types.py NER NER_NEW
if [[ -d "./NER_NEW" ]]; then
    rm -rf ./NER
fi
mv ./NER_NEW ./NER

# run twice
python consolidate-types.py NER NER_NEW
if [[ -d "./NER_NEW" ]]; then
    rm -rf ./NER
fi
mv ./NER_NEW ./NER

rm ./consolidate-types.py

# associate times and dates
cp ./helper_scripts/fix-split-durations-on-separate-lines.py ./

python fix-split-durations-on-separate-lines.py NER NER_NEW
if [[ -d "./NER_NEW" ]]; then
    rm -rf ./NER
fi
mv ./NER_NEW ./NER

rm ./fix-split-durations-on-separate-lines.py
