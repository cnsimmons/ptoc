#!/bin/bash

sub="$1"
run="$2"
raw_dir="/lab_data/behrmannlab/vlad/hemispace"
feat_dir="${raw_dir}/${sub}/ses-01/derivatives/fsl/loc/run-0${run}/1stLevel.feat"
highres2standard="${feat_dir}/reg/highres2standard.mat"

if [ ! -f "${highres2standard}" ]; then
    echo "Error: ${highres2standard} does not exist."
    exit 1
fi

echo "Contents of ${highres2standard}:"
cat ${highres2standard}

echo -e "\nMatrix information:"
avscale ${highres2standard}