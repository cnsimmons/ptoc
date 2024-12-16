#!/bin/bash

for i in {1001..1010}; do
    sub="sub-spaceloc$i"
    
    # Create derivatives directory if it doesn't exist
    mkdir -p /lab_data/behrmannlab/vlad/ptoc/${sub}/ses-01/derivatives
    
    # Run FLIRT
    flirt -in /lab_data/behrmannlab/vlad/hemispace/${sub}/ses-01/anat/${sub}_ses-01_T1w_brain.nii.gz \
         -ref $FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz \
         -out /lab_data/behrmannlab/vlad/ptoc/${sub}/ses-01/derivatives/anat2mni \
         -omat /lab_data/behrmannlab/vlad/ptoc/${sub}/ses-01/derivatives/anat2mni.mat
done