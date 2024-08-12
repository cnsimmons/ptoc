#!/bin/bash

#Directories
origDir="/Volumes/Zeus/DOC/Scripts/roiParcels/Mruczek_Parcels/subj_vol_all"
binaryDir="/Volumes/Zeus/DOC/Scripts/roiParcels/Mruczek_Parcels/Binary"
labelDir="/Volumes/Zeus/DOC/Parcels"

thresh=1

ROI=('V1' 'V2' 'V3' 'V3ab' 'PPC' 'APC')
vROI=('1' '3' '5' '16' '18' '22')
dROI=('2' '4' '6' '17' '19' '23')


for r in 0 1 2 3 4 5
do
	#echo $origDir/perc_VTPM_vol_roi${vROI[${r}]}_rh.nii.gz
	#echo $origDir/perc_VTPM_vol_roi${dROI[${r}]}_rh.nii.gz
	#combine left ROIs and save
	fslmaths $origDir/perc_VTPM_vol_roi${vROI[${r}]}_lh.nii.gz -add $origDir/perc_VTPM_vol_roi${dROI[${r}]}_lh.nii.gz -bin $binaryDir/l${ROI[${r}]}_Binary.nii.gz
	#combine right ROIs and save
	fslmaths $origDir/perc_VTPM_vol_roi${vROI[${r}]}_rh.nii.gz -add $origDir/perc_VTPM_vol_roi${dROI[${r}]}_rh.nii.gz -bin $binaryDir/r${ROI[${r}]}_Binary.nii.gz

	#Register to 2mm Anatomical
	flirt -in $binaryDir/l${ROI[${r}]}_Binary.nii.gz -ref /Volumes/Zeus/DOC/MNI152_T1_2mm_brain.nii.gz -out $labelDir/l${ROI[${r}]}_Binary.nii.gz  -applyxfm -init $labelDir/stand.mat -interp trilinear
	flirt -in $binaryDir/r${ROI[${r}]}_Binary.nii.gz -ref /Volumes/Zeus/DOC/MNI152_T1_2mm_brain.nii.gz -out $labelDir/r${ROI[${r}]}_Binary.nii.gz  -applyxfm -init $labelDir/stand.mat -interp trilinear

done
