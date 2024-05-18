#!/bin/bash

# Copyright 2021 VMware Tanzu Community Edition contributors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# set -o errexit
set -o nounset
set -o pipefail
set -o xtrace

rm -f train-v2.0.json
cp ./GOOD/train-v2.0.json ./

rm -f non_questions_dataset.csv
cp ./GOOD/non_questions_dataset.csv ./

rm -f questions_dataset.csv
cp ./GOOD/questions_dataset.csv ./

rm -f df_combined_dataset.csv
cp ./GOOD/df_combined_dataset.csv ./

# rm -rf results
rm -rf ./question_classifier_model
mkdir ./question_classifier_model
cp -r ./GOOD/question_classifier_model ./

rm -rf ./question_classifier_tokenizer
mkdir ./question_classifier_tokenizer
cp -r ./GOOD/question_classifier_tokenizer ./

cp ./GOOD/main.py ./
