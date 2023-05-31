#!/bin/bash

# move parcels into subjects' anatomical space.


rois2analyze="parcel_rOFA parcel_lOFA parcel_rFFA parcel_lFFA parcel_rpSTS parcel_rpSTS_1 parcel_rpSTS_2 parcel_rpSTS_3 parcel_rpSTS_4 parcel_lpSTS parcel_lpSTS_1 parcel_lpSTS_2 parcel_rOPA parcel_lOPA parcel_lOPA_1 parcel_lOPA_2 parcel_rPPA parcel_lPPA parcel_rRSC parcel_lRSC"

homeDir=/Volumes/Zeus/kidFaceSceneMotion/xScripts/xParcels

#first compute the transformation, stored in a .mat file, between CVS and MNI	
flirt -in $homeDir/forCharlie/cvs_avg35_reorient.nii.gz -ref $homeDir/forCharlie/MNI152_T1_1mm_brain.nii.gz -out $homeDir/forCharlie/cvs2anat -omat $homeDir/forCharlie/cvs2anat.mat #-bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12  -interp trilinear


#now apply that transformation to each ROI
for roi in $rois2analyze ; do
	echo $roi
	flirt -in $homeDir/${roi}.nii.gz -ref $homeDir/forCharlie/MNI152_T1_1mm_brain.nii.gz -out $homeDir/forCharlie/${roi}.nii.gz -applyxfm -init $homeDir/forCharlie/cvs2anat.mat -interp trilinear
	#binarize (since transformation can add weird weighting to voxels in image)
	fslmaths $homeDir/forCharlie/${roi}.nii.gz -bin $homeDir/forCharlie/${roi}.nii.gz
done

