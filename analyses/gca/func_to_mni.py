import os
import subprocess

def convert_to_mni(sub, run, input_file, output_file, highres2standard, mni_brain):
    if not os.path.isfile(input_file):
        print(f"Input file not found: {input_file}")
        return None

    if os.path.isfile(output_file):
        print(f"MNI file already exists: {output_file}")
        return output_file

    print(f"Converting to MNI space for {sub}, run {run}")
    cmd = [
        'applywarp',
        '--in=' + input_file,
        '--ref=' + mni_brain,
        '--out=' + output_file,
        '--premat=' + highres2standard,
        '--interp=spline'
    ]
    print("Running command:", ' '.join(cmd))
    subprocess.run(cmd, check=True)

    return output_file

# Setup parameters
raw_dir = "/lab_data/behrmannlab/vlad/hemispace"
mni_brain = os.path.join(os.environ['FSLDIR'], "data/standard/MNI152_T1_2mm_brain.nii.gz")

# Define subject and run (you can modify this to loop through multiple subjects/runs)
sub = "sub-025"
run = "3" #update to 3, 1 has run

# Define paths
feat_dir = f"{raw_dir}/{sub}/ses-01/derivatives/fsl/loc/run-0{run}/1stLevel.feat"
func_input = f"{feat_dir}/filtered_func_data.nii.gz"
highres2standard = f"{feat_dir}/reg/highres2standard.mat"
output_dir = f"{raw_dir}/{sub}/ses-01/derivatives/fsl/loc/run-0{run}/registered_data"
os.makedirs(output_dir, exist_ok=True)
output_file = f"{output_dir}/filtered_func_data_standard.nii.gz"

# Perform the conversion
result = convert_to_mni(sub, run, func_input, output_file, highres2standard, mni_brain)

if result:
    print(f"Conversion successful. Output: {result}")
else:
    print("Conversion failed.")