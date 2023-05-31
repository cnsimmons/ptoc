#!/bin/bash

mniBrain="/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz"

subj_list="1000 1001 1002 1003 1004 1005 1006 1007 1008 1009"
roi=('V1')
thresh=50

binaryDir="/Volumes/Zeus/DOC/Scripts/roiParcels/Mruczek_Parcels/Binary"
origDir="/Volumes/Zeus/DOC/Scripts/roiParcels/Mruczek_Parcels/LabledROIs"
dir="/Volumes/Zeus/DOC"


for s in $subj_list
do
	subjDir="$dir/DOC${s}/anat"
	roiDir="/Volumes/Zeus/DOC/DOC${s}/ROIs"

	#Create registration matrix from MNI to anatomy
	flirt -in $mniBrain -ref $subjDir/DOC${s}_anatomy_brain.nii.gz -omat $subjDir/stand2func.mat -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12	

	for i in 0 1
	do
		#Register Probabilistic files to anatomical
		flirt -in $origDir/${roi[${i}]}_lh.nii.gz -ref $subjDir/DOC${s}_anatomy_brain.nii.gz -out $roiDir/l${roi[${i}]}.nii.gz  -applyxfm -init $subjDir/stand2func.mat -interp trilinear
		flirt -in $origDir/${roi[${i}]}_rh.nii.gz -ref $subjDir/DOC${s}_anatomy_brain.nii.gz -out $roiDir/r${roi[${i}]}.nii.gz  -applyxfm -init $subjDir/stand2func.mat -interp trilinear

		#Binarize 
		#with hole filling
		fslmaths $roiDir/l${roi[${i}]}.nii.gz -thr $thresh -bin -fillh $roiDir/l${roi[${i}]}.nii.gz
		fslmaths $roiDir/r${roi[${i}]}.nii.gz -thr $thresh -bin -fillh $roiDir/r${roi[${i}]}.nii.gz
	done

done
