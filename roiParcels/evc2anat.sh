#!/bin/bash

# move parcels into subjects' anatomical space.

subj_list="SCM01"
# finished=
roi="parcel_lFFA parcel_lLOC parcel_lOFA parcel_lOPA parcel_lPFS parcel_lPPA parcel_lRSC parcel_lSTS parcel_raFFA parcel_rLOC parcel_rOFA parcel_rOPA parcel_rpFFA parcel_rPFS parcel_rPPA parcel_rpPFS parcel_rRSC parcel_rSTS"
contrast=(objects_scrambled scenes_objects)
parcel=(object scene)
index="0 1"

dir="/Volumes/Zeus/SCM"
parcelDir="$dir/scripts/roiParcels"

for s in $subj_list
do

mkdir $dir/$s/ROIs/parcelMats

subjDir="$dir/$s/anat"
mniBrain="/usr/local/fsl/data/standard/MNI152_T1_2mm_brain"

flirt -in $mniBrain -ref $subjDir/${s}_anatomy_brain.nii.gz -omat $subjDir/face.mat -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12

#flirt -in $parcelDir/faces_objects.nii -ref $mniBrain -omat $subjDir/face2.mat -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12

#convert_xfm -concat $subjDir/face1.mat -omat $subjDir/face.mat $subjDir/face2.mat

flirt -in $parcelDir/faces_objects.nii -ref $subjDir/${s}_anatomy_brain.nii.gz -out $subjDir/face -applyxfm -init $subjDir/face.mat -interp trilinear

mv $subjDir/*.mat $dir/$s/ROIs/parcelMats
mv $subjDir/face.nii.gz $dir/$s/ROIs

# Apply face xfm to other parcels
for i in $index
do
cp $dir/$s/ROIs/parcelMats/face.mat $subjDir/${parcel[$i]}.mat
#cp $dir/$s/ROIs/parcelMats/face1.mat $subjDir/${parcel[$i]}1.mat
#cp $dir/$s/ROIs/parcelMats/face2.mat $subjDir/${parcel[$i]}2.mat

#convert_xfm -concat $subjDir/${parcel[$i]}1.mat -omat $subjDir/${parcel[$i]}.mat $subjDir/${parcel[$i]}2.mat

flirt -in $parcelDir/${contrast[$i]}.nii -ref $subjDir/${s}_anatomy_brain.nii.gz -out $subjDir/${parcel[$i]} -applyxfm -init $subjDir/${parcel[$i]}.mat -interp trilinear

mv $subjDir/*.mat $dir/$s/ROIs/parcelMats
mv $subjDir/${parcel[$i]}.nii.gz $dir/$s/ROIs
done

for r in $roi
do


flirt -in $parcelDir/${r}.nii.gz -ref $subjDir/${s}_anatomy_brain.nii.gz -out $subjDir/${r} -applyxfm -init $dir/$s/ROIs/parcelMats/face.mat -interp trilinear

fslmaths $subjDir/${r}.nii.gz -div $subjDir/${r}.nii.gz $subjDir/${r}

mv $subjDir/${r}.nii.gz $dir/$s/ROIs

done
done
