#!/bin/bash

# Copyright 2021 VMware Tanzu Community Edition contributors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# set -o errexit
set -o nounset
set -o pipefail
set -o xtrace

rm -rf ./dogs
rm -rf ./models
rm -f dogs_classifier_model.pkl
rm -f boxer.jpg
rm -f german-shepherd.jpg
rm -f golden-retriever.jpg
