#!/bin/bash

# Set up main directories
hemispace_dir="/lab_data/behrmannlab/vlad/hemispace"
ptoc_dir="/lab_data/behrmannlab/vlad/ptoc"
mni_brain="${FSLDIR}/data/standard/MNI152_T1_2mm_brain.nii.gz"

# Array of subjects with proper formatting
subjects=(
    "sub-025"
    "sub-038"
    "sub-057"
    "sub-059"
    "sub-064"
    "sub-067"
    "sub-068"
    "sub-071"
    "sub-083"
    "sub-084"
    "sub-085"
    "sub-087"
    "sub-088"
    "sub-093"
    "sub-094"
    "sub-095"
    "sub-096"
    "sub-097"
    "sub-107"
)

#subjects=('sub-096')

# Array of runs (now including run 3)
runs=("01" "02" "03")

# Loop through each subject
for sub in "${subjects[@]}"; do
    echo "Processing subject: ${sub}"
    
    # Set up subject-specific directories
    # Note: Handling different session numbers might be needed
    hemi_sub_dir="${hemispace_dir}/${sub}/ses-01"
    ptoc_sub_dir="${ptoc_dir}/${sub}/ses-01"
    hemi_out_dir="${hemi_sub_dir}/derivatives"
    ptoc_out_dir="${ptoc_sub_dir}/derivatives"

    # Check for transformation matrix in ptoc
    ptoc_mat="${ptoc_out_dir}/anat2mni.mat"
    if [ ! -f "$ptoc_mat" ]; then
        echo "Transformation matrix not found at: ${ptoc_mat}"
        echo "Skipping subject ${sub}"
        continue
    fi

    # Create reg_standard directory if it doesn't exist
    mkdir -p "${hemi_out_dir}/reg_standard"

    # Loop through each run
    for run in "${runs[@]}"; do
        echo "Processing run: ${run}"

        # Define input and output filenames
        filtered_func="${hemi_out_dir}/fsl/loc/run-${run}/1stLevel.feat/filtered_func_data_reg.nii.gz"
        filtered_func_standard="${hemi_out_dir}/reg_standard/filtered_func_run-${run}_standard.nii.gz"

        # Check if input file exists
        if [ ! -f "$filtered_func" ]; then
            echo "Filtered func file not found at: ${filtered_func}"
            echo "Skipping run ${run} for subject ${sub}"
            continue
        fi

        # Check if output already exists
        if [ -f "$filtered_func_standard" ]; then
            echo "Standard space file already exists at: ${filtered_func_standard}"
            echo "Skipping run ${run} for subject ${sub}"
            continue
        fi

        # Perform the transformation
        echo "Registering filtered_func for ${sub}, Run ${run} to standard space"
        flirt \
            -in "$filtered_func" \
            -ref "$mni_brain" \
            -out "$filtered_func_standard" \
            -applyxfm \
            -init "$ptoc_mat" \
            -interp trilinear

        # Check if the transformation was successful
        if [ $? -eq 0 ]; then
            echo "Successfully converted Run ${run} to standard space"
            echo "Output saved to: ${filtered_func_standard}"
        else
            echo "Error during conversion for subject ${sub}, run ${run}"
            continue
        fi
    done
done

echo "Script execution completed for all subjects and runs."