#!/bin/bash
# Hand-run aCompCor FEAT on mind-0-16, for subjects NOT in the SLURM queue.
#
# Rules baked in:
#  - within a subject, runs go SEQUENTIALLY (never two runs of one subject at once
#    -- that's what killed sub-064 run-02)
#  - across subjects, run MAX_PARALLEL at a time
#  - skips any run whose filtered_func_data.nii.gz already exists (done)
#  - skips a run with no source .fsf (e.g. subjects with fewer runs)
#
# Excludes 025/038/057/059 by default -- those are in the SLURM queue.
# If you cancel those SLURM jobs, add them back to SUBS.
#
# Usage:  bash run_acompcor_local.sh

RAW=/lab_data/behrmannlab/vlad/hemispace
RUNS="1 2 3"
MAX_PARALLEL=3

# controls minus the 4 queued in SLURM (025 038 057 059) and sub-064 (hand-run separately)
SUBS="sub-067 sub-068 sub-071 sub-083 sub-084 sub-085 sub-087 sub-088 \
sub-093 sub-094 sub-095 sub-096 sub-097 sub-107"

run_subject () {
    local ss=$1
    for rn in $RUNS; do
        local fsf=$RAW/$ss/ses-01/derivatives/fsl/loc/run-0$rn/1stLevel_acompcor.fsf
        local done_file=$RAW/$ss/ses-01/derivatives/fsl/loc/run-0$rn/1stLevel_acompcor.feat/filtered_func_data.nii.gz
        if [ ! -f "$fsf" ]; then
            echo "[$ss run-0$rn] no .fsf, skip"
            continue
        fi
        if [ -f "$done_file" ]; then
            echo "[$ss run-0$rn] already done, skip"
            continue
        fi
        echo "[$ss run-0$rn] START $(date +%H:%M:%S)"
        feat "$fsf"
        echo "[$ss run-0$rn] END   $(date +%H:%M:%S)"
    done
}

n=0
for ss in $SUBS; do
    run_subject "$ss" &          # one subject in background (its runs are sequential inside)
    n=$((n+1))
    if [ $((n % MAX_PARALLEL)) -eq 0 ]; then
        wait                      # wait for this batch of subjects before starting the next
    fi
done
wait
echo "ALL DONE $(date +%H:%M:%S)"