#!/bin/bash

# First, let's create a shell script (process_reg_standard.sh):

# Setup FSL
. /etc/fsl/5.0/fsl.sh  # adjust path as needed

# Base directory
RAW_DIR="/lab_data/behrmannlab/vlad/hemispace"

# Function to process one run
process_run() {
    sub=$1
    run=$2
    
    echo "Processing ${sub} run-${run}"
    
    # Setup directories
    FEAT_DIR="${RAW_DIR}/${sub}/ses-01/derivatives/fsl/loc/run-${run}/1stLevel.feat"
    REG_STD_DIR="${FEAT_DIR}/reg_standard"
    
    # Check if reg_standard already exists
    if [ ! -d "${REG_STD_DIR}" ]; then
        echo "Creating reg_standard directory for ${sub} run-${run}"
        mkdir -p "${REG_STD_DIR}"
        
        # Transform mean_func to 2mm MNI space
        flirt -in "${FEAT_DIR}/mean_func" \
              -ref "${FSLDIR}/data/standard/MNI152_T1_2mm_brain" \
              -out "${REG_STD_DIR}/mean_func" \
              -init "${FEAT_DIR}/reg/example_func2standard.mat" \
              -applyxfm
        
        # Create mask
        fslmaths "${REG_STD_DIR}/mean_func" \
                 -thr 100 -bin \
                 "${REG_STD_DIR}/mask"
                 
        echo "Completed ${sub} run-${run}"
    else
        echo "reg_standard already exists for ${sub} run-${run}"
    fi
}

# List of subjects to process
subjects=("sub-084")

# Process each subject's run-03
for sub in "${subjects[@]}"; do
    process_run "$sub" "01"
done

echo "All processing complete"