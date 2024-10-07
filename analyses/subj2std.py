import os
import subprocess
import argparse

def transform_func_to_mni(func_file, anat2mni_mat, mni_brain, output_file):
    if not os.path.isfile(output_file):
        print(f"Registering functional data to MNI space: {func_file}")
        subprocess.run([
            'flirt',
            '-in', func_file,
            '-ref', mni_brain,
            '-out', output_file,
            '-applyxfm',
            '-init', anat2mni_mat,
            '-interp', 'trilinear'
        ], check=True)
        print(f"Output saved to: {output_file}")
    else:
        print(f"MNI-space functional data already exists: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Transform functional data to MNI space")
    parser.add_argument("--subject", required=True, help="Subject ID (e.g., sub-038)")
    parser.add_argument("--task", required=True, help="Task name (e.g., loc)")
    parser.add_argument("--runs", nargs='+', required=True, help="Run numbers (e.g., 1 2 3)")
    args = parser.parse_args()

    study_dir = "/lab_data/behrmannlab/vlad/ptoc"
    raw_dir = "/lab_data/behrmannlab/vlad/hemispace"
    mni_brain = os.path.join(os.environ['FSLDIR'], "data/standard/MNI152_T1_2mm_brain.nii.gz")

    sub_dir = f"{study_dir}/{args.subject}/ses-01"
    out_dir = f"{sub_dir}/derivatives"
    anat2mni_mat = f"{out_dir}/anat2mni.mat"

    if not os.path.isfile(anat2mni_mat):
        print(f"Transformation matrix not found for {args.subject}. Exiting...")
        return

    for run in args.runs:
        func_file = f'{raw_dir}/{args.subject}/ses-01/derivatives/fsl/{args.task}/run-0{run}/1stLevel.feat/filtered_func_data.nii.gz'
        mni_func_file = f'{out_dir}/func_mni/run-0{run}_filtered_func_data_mni.nii.gz'
        os.makedirs(f'{out_dir}/func_mni', exist_ok=True)
        
        transform_func_to_mni(func_file, anat2mni_mat, mni_brain, mni_func_file)

    print(f"Transformation to MNI space completed for {args.subject}, task {args.task}, runs {', '.join(args.runs)}.")

if __name__ == "__main__":
    main()