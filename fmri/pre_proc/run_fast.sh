module load fsl-6.0.3
RAW=/lab_data/behrmannlab/vlad/hemispace

for sub in $(awk -F',' 'NR>1 && $9=="control"{print $1}' /user_data/csimmon2/git_repos/ptoc/sub_info.csv); do
    sub=${sub/#sub-/}; sub="sub-${sub}"
    brain=$RAW/$sub/ses-01/anat/${sub}_ses-01_T1w_brain.nii.gz
    out=$RAW/$sub/ses-01/anat/${sub}_fast
    if [ -f "${out}_pve_0.nii.gz" ]; then echo "skip $sub (done)"; continue; fi
    if [ ! -f "$brain" ]; then echo "MISSING brain: $sub"; continue; fi
    echo "FAST $sub"
    fast -t 1 -n 3 -o "$out" "$brain"
done