#!/bin/bash

# Setup FSL
. /etc/fsl/5.0/fsl.sh

# Base directory
RAW_DIR="/lab_data/behrmannlab/vlad/hemispace"

process_run() {
    sub=$1
    run=$2
    
    echo "Processing ${sub} run-${run}"
    
    # Setup directories
    FEAT_DIR="${RAW_DIR}/${sub}/ses-01/derivatives/fsl/loc/run-${run}/1stLevel.feat"
    REG_STD_DIR="${FEAT_DIR}/reg_standard"
    REG_DIR="${FEAT_DIR}/reg"
    
    if [ -d "${FEAT_DIR}" ]; then
        mkdir -p "${REG_STD_DIR}"
        
        # First concatenate the transformations
        echo "Creating combined transformation..."
        convert_xfm -omat ${REG_DIR}/func2standard.mat \
                   -concat ${REG_DIR}/example_func2standard.mat ${REG_DIR}/example_func2highres.mat
        
        # Transform filtered_func_data to standard space using the combined transform
        echo "Transforming filtered_func_data to standard space..."
        applywarp --in="${FEAT_DIR}/filtered_func_data" \
                 --ref="${FSLDIR}/data/standard/MNI152_T1_2mm_brain" \
                 --out="${REG_STD_DIR}/filtered_func_data_standard" \
                 --premat="${REG_DIR}/func2standard.mat" \
                 --interp=spline
        
        # Verify the output exists and has reasonable dimensions
        if [ -f "${REG_STD_DIR}/filtered_func_data_standard.nii.gz" ]; then
            dims=$(fslinfo "${REG_STD_DIR}/filtered_func_data_standard.nii.gz" | grep ^dim[123])
            echo "Output dimensions: ${dims}"
        else
            echo "Warning: Output file not created"
        fi
        
        echo "Completed ${sub} run-${run}"
    else
        echo "Directory not found: ${FEAT_DIR}"
    fi
}

# Test on one subject first
sub="sub-064"
run="03"
process_run "$sub" "$run"