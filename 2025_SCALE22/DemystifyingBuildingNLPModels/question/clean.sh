#!/bin/bash

# set -o errexit
set -o nounset
set -o pipefail
set -o xtrace

rm -f train-v2.0.json
rm -f non_questions_dataset.csv
rm -f questions_dataset.csv
rm -f df_combined_dataset.csv

rm -rf results
rm -rf question_classifier_model
rm -rf question_classifier_tokenizer
