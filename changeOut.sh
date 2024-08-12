#!/bin/sh

#  Copies firstLevel and highlevels from one run to the others
#  
#
#  Created by VA + JK on 8/13/19.
#

exp="ptoc" #experiment name
ogSub="004" #which sub to copy from
subj_list="007 025 038 057 059" #which subs to copy to (add as many as you have)
subj_list="064 066 067 068 069 071 074 075 076 077 078 079 083 084 085 087 088 089 
090 091 092 093 094 095 096 097 103 106 107 108 109 hemispace1004 hemispace1007 hemispace1006"
subj_list="hemispace1006 hemispace1004 hemispace1007"

runs=("1" "2" "3") #runs to copy

cond="loc" #which condition to copy
suf="" #any suffix to add
sesh="01" #session to copy


dataDir=/lab_data/behrmannlab/vlad/${exp}

###############################
ogDir=$dataDir/sub-${ogSub}/ses-${sesh}/derivatives/fsl
for sub in $subj_list
do
	
	subjDir=$dataDir/sub-${sub}/ses-${sesh}/derivatives/fsl


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


			
			continue

		done

		#COMMENTING THIS OUT FOR NOW BECAUSE YOU DONT HAVE A HIGHLEVEL YET!!
		cp ${ogRun}/HighLevel${suf}.fsf $runDir/HighLevel${suf}.fsf #copies fsf from run 1 into the other runs (cp = copy)
		sed -i "s/${ogSub}/${sub}/g" $runDir/HighLevel${suf}.fsf


	echo $s
	done
done

echo "Dunzo!!!"