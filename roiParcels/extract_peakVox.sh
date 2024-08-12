n=50
roi="rLO"
copeNum="2"
  
nVox=`fslstats ${roiDir}/${r}.nii.gz -V`
dir="/Volumes/Zeus/DOC"
funcDir="$subjDir/DOTS_func/HighLevel_4Runs.gfeat"
roiDir="$subjDir/ROIs"
nVox=`fslstats ${roiDir}/${r}.nii.gz -V`
nVox=(${nVox[@]})

topPerc=$(calc $n/$nVox)

topPerc=$(calc $topPerc*100)

fslmaths $funcDir/cope${copeNum}.feat/stats/zstat1.nii.gz -mul $roiDir/$r.nii.gz $roiDir/${r}_func.nii.gz

fslmaths $funcDir/cope${copeNum}.feat/stats/zstat1.nii.gz -mul $roiDir/$r.nii.gz /Volumes/Zeus/DOC/DOTS1003/ROIs/rLO_func.nii.gz 
thresh=`fslstats $roiDir/$r_func.nii.gz -P $topPerc`
  
fslmaths $roiDir/${r}_func.nii.gz -thr $thresh -bin $roiDir/${r}_peak.nii.gz
