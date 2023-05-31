#!/bin/sh

#  Copies firstLevel from one run to the others
#  
#
#  Created by VA + JK on 8/13/19.
#


subj_list="spaceloc1001 spaceloc1002 spaceloc1003 spaceloc1004 spaceloc1005 spaceloc1006 spaceloc1007 spaceloc1008 spaceloc1009 spaceloc1010 spaceloc1011 spaceloc1012 spaceloc2013 spaceloc2014 spaceloc2015 spaceloc2016 spaceloc2017 spaceloc2018"
#subj_list="025 038 057 059 064 067 068 071 083 084 085 087 088 093 094 095 096 097 103 104 105 106 107 hemispace1001 hemispace1002 hemispace1003 hemispace2001 hemispace2002 hemispace2003"
#subj_list="108 109"
runs=("1" "2" "3" "4" "5" "6" "7" "8")
runs=("1" "2")
ogSub="spaceloc1001"
exp="bwoc"
#exp="docnet"
#cond="spaceloc depthloc distloc toolloc"
cond="toolloc"
suf=""
sesh="01"


dataDir=/lab_data/behrmannlab/vlad/${exp}

###############################
ogDir=$dataDir/sub-${ogSub}/ses-${sesh}/derivatives/fsl
for sub in $subj_list
do
	
	subjDir=$dataDir/sub-${sub}/ses-${sesh}/derivatives/fsl

#cd $subjDir

	for cc in $cond; do
		runDir=$subjDir/${cc}
		ogRun=$ogDir/${cc}

		
		echo ${ogRun}/run-01/1stLevel${suf}.fsf
		for r in "${runs[@]}";
		do

			mkdir -p $runDir/run-0${r}

			cp ${ogRun}/run-01/1stLevel${suf}.fsf $runDir/run-0${r}/1stLevel${suf}.fsf #copies fsf from run 1 into the other runs (cp = copy)

			sed -i "s/${ogSub}/${sub}/g" $runDir/run-0${r}/1stLevel${suf}.fsf #change subject
			sed -i "s/run-01/run-0${r}/g" $runDir/run-0${r}/1stLevel${suf}.fsf #change run for file and output
			sed -i "s/run1/run${r}/g" $runDir/run-0${r}/1stLevel${suf}.fsf #change run for file and output
			sed -i "s/Run1/Run${r}/g" $runDir/run-0${r}/1stLevel${suf}.fsf #change run for covs

			#copy catloc files
			
			continue

		done

		cp ${ogRun}/HighLevel${suf}.fsf $runDir/HighLevel${suf}.fsf #copies fsf from run 1 into the other runs (cp = copy)
		sed -i "s/${ogSub}/${sub}/g" $runDir/HighLevel${suf}.fsf

		#cp ${ogRun}/HighLevel_odd.fsf $runDir/HighLevel_odd.fsf #copies fsf from run 1 into the other runs (cp = copy)
		#sed -i "s/${ogSub}/${sub}/g" $runDir/HighLevel_odd.fsf

		#cp ${ogRun}/HighLevel_even.fsf $runDir/HighLevel_even.fsf #copies fsf from run 1 into the other runs (cp = copy)
		#sed -i "s/${ogSub}/${sub}/g" $runDir/HighLevel_even.fsf

		#cp ${ogRun}/HighLevel_roi_2runs.fsf $runDir/HighLevel_roi_2runs.fsf #copies fsf from run 1 into the other runs (cp = copy)
		#sed -i "s/${ogSub}/${sub}/g" $runDir/HighLevel_roi_2runs.fsf

	#rm $runDir/HighLevel_roi_2run.fsf 
	#rm $runDir/HighLevel_roi_run.fsf 
	#cp ${ogRun}/HighLevel_6Runs.fsf $runDir/HighLevel_6Runs.fsf #copies fsf from run 1 into the other runs (cp = copy)
	#sed -i "s/${ogSub}/${sub}/g" $runDir/HighLevel_6Runs.fsf

	echo $s
	done
done

echo "Dunzo!!!"