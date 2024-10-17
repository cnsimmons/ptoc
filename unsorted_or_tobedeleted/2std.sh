#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

raw_dir="/lab_data/behrmannlab/vlad/hemispace"
sub="$1"
run="$2"
feat_dir="${raw_dir}/${sub}/ses-01/derivatives/fsl/loc/run-0${run}/1stLevel.feat"
output_dir="${raw_dir}/${sub}/ses-01/derivatives/fsl/loc/run-0${run}/registered_data"

mkdir -p ${output_dir}

standard_brain="/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz"
func_input="${feat_dir}/filtered_func_data_reg.nii.gz"  # Changed this line
highres2standard="${feat_dir}/reg/highres2standard.mat"

# Check input files
for file in "${func_input}" "${highres2standard}" "${standard_brain}"; do
    if [ ! -f "${file}" ]; then
        echo "Error: ${file} does not exist."
        exit 1
    fi
done

# Check registration matrix
echo "Checking highres2standard matrix:"
avscale ${highres2standard} | grep 'Translations' -A 1

# Check standard space
echo "Checking standard space:"
fslinfo ${standard_brain} | grep -E 'dim1|dim2|dim3|pixdim1|pixdim2|pixdim3'

# Apply transformation using applywarp
applywarp --in=${func_input} \
          --ref=${standard_brain} \
          --out=${output_dir}/filtered_func_data_standard.nii.gz \
          --premat=${highres2standard} \
          --interp=spline

if [ -f "${output_dir}/filtered_func_data_standard.nii.gz" ]; then
    echo "Registration complete. Output: ${output_dir}/filtered_func_data_standard.nii.gz"
    # Check output space
    echo "Checking output space:"
    fslinfo ${output_dir}/filtered_func_data_standard.nii.gz | grep -E 'dim1|dim2|dim3|pixdim1|pixdim2|pixdim3'
else
    echo "Error: Output file was not created."
    exit 1
fi

# Optional: Visual check
fsleyes ${output_dir}/filtered_func_data_standard.nii.gz ${standard_brain}