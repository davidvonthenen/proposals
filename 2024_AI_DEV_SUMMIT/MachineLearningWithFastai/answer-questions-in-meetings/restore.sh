#!/bin/bash

# Copyright 2021 VMware Tanzu Community Edition contributors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# set -o errexit
set -o nounset
set -o pipefail
set -o xtrace

# tmp audio file
rm -f output.mp3

# rm -rf results
rm -rf ./question_classifier_model
mkdir ./question_classifier_model
cp -r ./SAVE/question_classifier_model ./

rm -rf ./question_classifier_tokenizer
mkdir ./question_classifier_tokenizer
cp -r ./SAVE/question_classifier_tokenizer ./
