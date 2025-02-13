#!/bin/bash

# Base directories
STUDY_DIR="/lab_data/behrmannlab/vlad/ptoc"
RESULTS_DIR="/user_data/csimmon2/git_repos/ptoc/results"

# Create output directory for FreeSurfer space files
FS_OUT_DIR="${RESULTS_DIR}/freesurfer_space"
mkdir -p ${FS_OUT_DIR}

# Check FreeSurfer setup
if [ -z "${FREESURFER_HOME}" ]; then
    echo "Error: FREESURFER_HOME not set. Please source FreeSurfer's setup file."
    exit 1
fi

# Use standard FreeSurfer MNI152 registration file
REG_FILE="$FREESURFER_HOME/average/mni152.register.dat"

if [ ! -f "$REG_FILE" ]; then
    echo "Error: Standard registration file not found: $REG_FILE"
    exit 1
fi

# Convert group average files
GROUP_AVG_DIR="${RESULTS_DIR}/group_averages"
for roi in pIPS LO; do
    for hemi in left right; do
        for analysis in fc ppi; do
            input_file="${GROUP_AVG_DIR}/${roi}_${hemi}_${analysis}_avg.nii.gz"
            if [ -f "$input_file" ]; then
                output_file="${FS_OUT_DIR}/group_${roi}_${hemi}_${analysis}_avg_fs.mgz"
                echo "Converting group average: ${input_file}"
                
                mri_vol2vol \
                    --mov ${input_file} \
                    --targ $SUBJECTS_DIR/fsaverage/mri/orig.mgz \
                    --reg ${REG_FILE} \
                    --o ${output_file}
                
                if [ $? -eq 0 ]; then
                    echo "Success: ${output_file}"
                else
                    echo "Error converting ${input_file}"
                fi
            fi
        done
    done
done

echo "Conversion complete. Files are in: ${FS_OUT_DIR}"