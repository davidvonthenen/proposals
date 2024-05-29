#!/bin/bash

# Copyright 2021 VMware Tanzu Community Edition contributors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# set -o errexit
set -o nounset
set -o pipefail
set -o xtrace

cp -r ./SAVE/dogs ./

cp ./SAVE/dogs_classifier_model.pkl ./
cp ./SAVE/boxer.jpg ./
cp ./SAVE/german-shepherd.jpg ./
cp ./SAVE/golden-retriever.jpg ./
