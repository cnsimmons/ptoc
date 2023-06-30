#!/bin/bash

#Registers OFC ROI to each 1stLevel for PPI analysis
origDir="/Volumes/Zeus/DOC/Scripts/roiParcels/ProbAtlas_v4"
dir="/Volumes/Zeus/DOC"

subj_list="1002"
ROIs=('OFC')

thresh=90

runNum="1 2 3 4 5"

for s in $subj_list
do
	subjDir="$dir/DOTS${s}"
	anat="$subjDir/anat/DOTS${s}_anatomy_brain.nii.gz"
	roiDir="$dir/DOTS${s}/ROIs"

	#flirt -in $mniBrain -ref $subjDir/DOTS${s}_anatomy_brain.nii.gz -omat $subjDir/stand2func.mat -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12	

	flirt -in $origDir/lOFC.nii.gz -ref $subjDir/anat/DOTS${s}_anatomy_brain.nii.gz -out $roiDir/lOFC_prob.nii.gz  -applyxfm -init $subjDir/anat/stand2func.mat -interp trilinear
	flirt -in $origDir/rOFC.nii.gz -ref $subjDir/anat/DOTS${s}_anatomy_brain.nii.gz -out $roiDir/rOFC_prob.nii.gz  -applyxfm -init $subjDir/anat/stand2func.mat -interp trilinear

done


#/Volumes/Zeus/MA-MRI/Scripts/roiParcels/ProbAtlas_v4/lOFC.nii.gz
#/Volumes/Zeus/DOC/Scripts/roiParcels/ProbAtlas_v4/lOFC.nii.gz