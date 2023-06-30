#!/bin/bash

mniBrain="/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz"

subj_list="1000 1001 1002 1003 1004 1005 1006 1007 1008 1009"
#subj_list="2001"
roi=('V3a' 'V3b' 'IPS0' 'IPS1' 'IPS4' 'IPS5')
roiNum=('16' '17' '18' '19' '22' '23')


binaryDir="/Volumes/Zeus/DOC/Scripts/roiParcels/Mruczek_Parcels/Binary"
origDir="/Volumes/Zeus/DOC/Scripts/roiParcels/Mruczek_Parcels/subj_vol_all"
dir="/Volumes/Zeus/DOC"


for s in $subj_list
do
	subjDir="$dir/DOC${s}/anat"
	roiDir="$dir/DOC${s}/ROIs/"
	mkdir $roiDir
	roiDir="$dir/DOC${s}/ROIs/Parcels/"
	mkdir $roiDir
	

	#Create registration matrix from MNI to anatomy
	flirt -in $mniBrain -ref $subjDir/DOC${s}_anatomy_brain.nii.gz -omat $subjDir/stand2func.mat -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12	

	for i in 0 1 2 3 4 5
	do
		echo $origDir/perc_VTPM_vol_roi${roiNum[${i}]}_lh.nii.gz
		#Register Probabilistic files to anatomical
		flirt -in $origDir/perc_VTPM_vol_roi${roiNum[${i}]}_lh.nii.gz -ref $subjDir/DOC${s}_anatomy_brain.nii.gz -out $roiDir/l${roi[${i}]}_prob.nii.gz  -applyxfm -init $subjDir/stand2func.mat -interp trilinear
		flirt -in $origDir/perc_VTPM_vol_roi${roiNum[${i}]}_rh.nii.gz -ref $subjDir/DOC${s}_anatomy_brain.nii.gz -out $roiDir/r${roi[${i}]}_prob.nii.gz  -applyxfm -init $subjDir/stand2func.mat -interp trilinear

		#Register binary files to anatomical
		flirt -in $binaryDir/perc_VTPM_vol_roi${roiNum[${i}]}_lh.nii.gz -ref $subjDir/DOC${s}_anatomy_brain.nii.gz -out $roiDir/l${roi[${i}]}_binary.nii.gz  -applyxfm -init $subjDir/stand2func.mat -interp trilinear
		flirt -in $binaryDir/perc_VTPM_vol_roi${roiNum[${i}]}_rh.nii.gz -ref $subjDir/DOC${s}_anatomy_brain.nii.gz -out $roiDir/r${roi[${i}]}_binary.nii.gz  -applyxfm -init $subjDir/stand2func.mat -interp trilinear
	done

done

echo "DUNZO!!??!"