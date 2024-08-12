import subprocess
import os
import pdb

#ADD "spaceloc1001",
subj_list=["spaceloc1001","spaceloc1002","spaceloc1003","spaceloc1004","spaceloc1005", "spaceloc1006",
"spaceloc1007","spaceloc1008","spaceloc1009","spaceloc1010", "spaceloc1011","spaceloc1012", 
"spaceloc2013", "spaceloc2014", "spaceloc2015", "spaceloc2016", "spaceloc2017", "spaceloc2018"]



#subj_list = list(range(2000,2020))

exp="bwoc"
cond=["spaceloc", "depthloc", "distloc","toolloc"]
cond = ["toolloc"]
#cond=["FBOSS_func"]
loc_suf=""

#Rois
roi=["V3ab", "PPC", "APC", "V4", "LO", "PFS"]
roi=["PPC", "APC", "LO", "PFS"]


#Specify copes (contrasts) of interest
#the number of nested lists corresponds to the number of conidionts (e.g., spaceloc, depthloc etc)
#the number of copes in each list corresponds to the number of ROIs (e.g., PPC, APC.. etc)
cope_num=[[1, 1, 1, 2, 2, 2], [1, 1, 1, 2, 2, 2], [1, 1, 1, 2, 2, 2], [1,1,1,5,5,5]]
cope_num=[[1, 1,  2, 2], [1, 1, 2, 2], [1, 1, 2, 2], [1,1,5,5]]
cope_num=[[5, 5,5,5]]


exp_dir=f"/lab_data/behrmannlab/vlad/{exp}"

for ss in subj_list:
    sub_dir = f"{exp_dir}/sub-{ss}/ses-01/derivatives"
    #sub_dir = f"{exp_dir}/MAMRI{ss}/"
    roi_dir = f'{sub_dir}/rois'

    os.makedirs(f'{roi_dir}/data', exist_ok=True)

    for cc_num, cc in enumerate(cond):
        func_dir = f'{sub_dir}/fsl/{cc}/HighLevel{loc_suf}.gfeat'
        #func_dir = f'{sub_dir}/{cc}/HighLevel{loc_suf}.gfeat'

        for rr in range(0,len(roi)):

            for lr in ["l", "r"]:
                
                #Maybe check if ROI and data exist before creating it
                print(ss, lr, cc, roi[rr], cope_num[cc_num][rr])

                #define ROI directory
                roi_nifti = f'{roi_dir}/{lr}{roi[rr]}_{cc}.nii.gz'
                cope_dir = f'{func_dir}/cope{cope_num[cc_num][rr]}.feat'
                #print(cope_dir)

                #Extract ROI by multiplying the cluster-correct mask by broad anatomical parcel

                bash_cmd = f'fslmaths {cope_dir}/cluster_mask_zstat1.nii.gz -mul {roi_dir}/parcels/{lr}{roi[rr]}.nii.gz -bin {roi_nifti}'
                print(bash_cmd)
                subprocess.run(bash_cmd.split(), check=True)

                #check if the mask is empty, delete if it is
                bash_cmd = f'fslstats {roi_nifti} -V'
                vox_num = subprocess.run(bash_cmd.split(),check=True, capture_output=True, text=True) #this subprocess captures the output

                if vox_num.stdout[0] == "0":
                    os.remove(roi_nifti)

                else: #else save functional data from mask
                    roi_data = f'{roi_dir}/data/{lr}{roi[rr]}_{cc}'

                    bash_cmd = f'fslmeants -i {cope_dir}/stats/zstat1.nii.gz -m {roi_nifti} -o {roi_data}.txt --showall --transpose'
                    subprocess.run(bash_cmd.split(), check=True)







