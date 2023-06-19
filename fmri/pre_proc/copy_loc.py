import sys
curr_dir = '/user_data/vayzenbe/GitHub_Repos/hemi_bwoc'
sys.path.append(curr_dir)
import os
import shutil
import pandas as pd
import numpy as np
import pdb


source_dir = '/lab_data/behrmannlab/hemi/Raw'
target_dir = '/lab_data/behrmannlab/vlad/hemispace'
copy_locs = False

og_sub = 'hemispace2001'
conds = ['Face', 'House', 'Object', 'Scramble', 'Word']

ses = '01'

runs = [1,2,3]
#load sub data
sub_data = pd.read_csv(f'{curr_dir}/ptoc_sub_info.csv')

def create_cov(src_dir, dest_dir, sub,ses,run):
    cov_file = f'{src_dir}/{sub}_{ses}_task-loc_run-0{run}_events.tsv'

    cov = pd.read_csv(cov_file, sep='\t')

    conds = cov.block_type.unique().tolist()

    for cond in conds:
        dest_cov = f'{dest_dir}/catloc_{sub[-3:]}_run-0{run}_{cond}.txt'
        curr_cov = cov[cov['block_type'] == cond].iloc[:,0:2]
        curr_cov['value'] = np.zeros((len(curr_cov))) + 1

        curr_cov.to_csv(dest_cov, index= False, header =False, sep = '\t')

message_list = []
for sub, ses in zip(sub_data['ID'], sub_data['Session']):

    print('Copying data for subject: ', sub, ses)
    #copy anat
    #make path to anat
    anat_file = f'{source_dir}/{sub}/{ses}/anat/{sub}_{ses}_T1w.nii.gz'
    target_anat = f'{target_dir}/{sub}/ses-01/anat/{sub}_ses-01_T1w.nii.gz'


    #copy anat
    if os.path.exists(anat_file):
        #create target dir
        os.makedirs(f'{target_dir}/{sub}/ses-01/anat', exist_ok=True)
        shutil.copy(anat_file, target_anat)
    else:
        print('No anat file', sub)
        message_list.append(f'No anat file {sub} {ses}')

    #copy functionals
    for run in runs:

        #make path to func
        func_file = f'{source_dir}/{sub}/{ses}/func/{sub}_{ses}_task-loc_run-0{run}_bold.nii.gz'

        #rename to ses-01
        new_func_file = f'{target_dir}/{sub}/ses-01/func/{sub}_ses-01_task-loc_run-0{run}_bold.nii.gz'
        #check if file exists
        if os.path.exists(func_file):
            #create target dir
            os.makedirs(f'{target_dir}/{sub}/ses-01/func', exist_ok=True)
            os.makedirs(f'{target_dir}/{sub}/ses-01/covs', exist_ok=True)
            shutil.copy(func_file, new_func_file)

            create_cov(f'{source_dir}/{sub}/{ses}/func', f'{target_dir}/{sub}/ses-01/covs', sub,ses, run)

        else:
            print('No func file', sub, ses, run)
            message_list.append(f'No func file {sub} {ses} {run}')
        #copy catloc cov from og_sub directory
        #Face, House, Object, Scramble, word
        
#save message list as txt file
#convert message list to dataframe
message_df = pd.DataFrame(message_list) 
message_df.to_csv(f'{curr_dir}/message_list.txt', index=False, header=False)


        


if copy_locs:
    #copy folder from source to target
    for sub, ses in zip(sub_data['ID'], sub_data['Session']):
        source_path = f'{source_dir}/{sub}/{ses}'
        target_path = f'{target_dir}/{sub}/ses-01'
        #print(os.path.exists(source_path), source_path)
        #check if source dir exists
        if os.path.exists(source_path):

            #make target path
            os.makedirs(target_path, exist_ok=True)



            #copy sub folder from source to target
            print(source_path, target_path)

            shutil.copytree(source_path, target_path,dirs_exist_ok=True)

            #rename folder to ses-01
            #os.rename(f'{target_dir}/{sub}/{ses}', f'{target_dir}/{sub}/ses-01')
