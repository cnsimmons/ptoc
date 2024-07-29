#this script for bash, the jupyter notebook is for testing

curr_dir = f'/user_data/csimmon2/git_repos/ptoc'

import sys
sys.path.insert(0,curr_dir)
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import scipy.stats as stats
import scipy
import statsmodels.api as s
from sklearn import metrics

import pdb
import ptoc_params as params

from plotnine import *
#from plotnine import ggplot, aes, geom_point





#hide warnings
import warnings
warnings.filterwarnings('ignore')

#load additional libraries
from nilearn import image, plotting, input_data, glm
from nilearn.input_data import NiftiMasker
import nibabel as nib
import statsmodels.api as sm
from nilearn.datasets import load_mni152_brain_mask, load_mni152_template


data_dir = params.data_dir
results_dir = params.results_dir
fig_dir = params.fig_dir
raw_dir = params.raw_dir

sub_info = params.sub_info
task_info = params.task_info

suf = params.suf
rois = params.rois
hemis = params.hemis

#load subject info
sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')

#mni = load_mni152_brain_mask()






'''exp info'''
subs = ['sub-064']  # Run for one subject initially
#subs = sub_info['sub'].tolist()

#Just controls
subs = sub_info[sub_info['group'] == 'control']['sub'].tolist()
study = 'ptoc'
data_dir = 'hemispace'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
out_dir = f'{study_dir}/derivatives/fc'
results_dir = '/user_data/csimmon2/GitHub_Repos/ptoc/results'
exp = ''
rois = ['LO']  # Run for one ROI initially
control_tasks = ['loc']
file_suf = ''

'''scan params'''
tr = 1
vols = 321

whole_brain_mask = load_mni152_brain_mask()
mni = load_mni152_template()
brain_masker = NiftiMasker(whole_brain_mask, smoothing_fwhm=0, standardize=True)

'''run info'''
run_num = 3
runs = list(range(1, run_num + 1))
run_combos = []

for rn1 in range(1, run_num + 1):
    for rn2 in range(rn1 + 1, run_num + 1):
        run_combos.append([rn1, rn2])
        
        



#Extract ROI coordinates
def extract_roi_coords():
    """
    Define ROIs
    """
    parcels = ['V1', 'aIPS', 'PFS', 'pIPS', 'LO']
    subs = sub_info[sub_info['group'] == 'control']['sub'].tolist()

    for ss in subs:
        print(f'Processing subject: {ss}')
        sub_dir = f'{study_dir}/{ss}/ses-01'
        roi_dir = f'{sub_dir}/derivatives/rois'
        os.makedirs(f'{roi_dir}/spheres', exist_ok=True)
        
        exp_dir = f'{sub_dir}/derivatives/fsl'
        parcel_dir = f'{roi_dir}/parcels'
        roi_coords = pd.DataFrame(columns=['index', 'task', 'roi', 'x', 'y', 'z'])
        
        for rcn, rc in enumerate(run_combos):
            roi_runs = [ele for ele in runs if ele not in rc]
            
            #load each run
            all_runs = []
            for rn in roi_runs:
                curr_run_path = f'{exp_dir}/loc/run-0{rn}/1stLevel.feat/stats/zstat3_reg.nii.gz'
                if os.path.exists(curr_run_path):
                    curr_run = image.load_img(curr_run_path)
                    all_runs.append(curr_run)
                else:
                    print(f'File does not exist: {curr_run_path}')
            
            mean_zstat = image.mean_img(all_runs)
            affine = mean_zstat.affine

            for pr in parcels:
                roi_path = f'{parcel_dir}/{pr}.nii.gz'
                if os.path.exists(roi_path):
                    roi = image.load_img(roi_path)
                    roi = image.math_img('img > 0', img=roi)

                    coords = plotting.find_xyz_cut_coords(mean_zstat, mask_img=roi, activation_threshold=0.99)
                    
                    masked_stat = image.math_img('img1 * img2', img1=roi, img2=mean_zstat)
                    masked_stat = image.get_data(masked_stat)
                    np_coords = np.where(masked_stat == np.max(masked_stat))
                    
                    curr_coords = pd.Series([rcn, 'loc', pr] + coords, index=roi_coords.columns)
                    roi_coords = roi_coords.append(curr_coords, ignore_index=True)

                    # control task ROI
                    control_zstat_path = f'{exp_dir}/loc/HighLevel.gfeat/cope3.feat/stats/zstat1.nii.gz'
                    if os.path.exists(control_zstat_path):
                        control_zstat = image.load_img(control_zstat_path)
                        coords = plotting.find_xyz_cut_coords(control_zstat, mask_img=roi, activation_threshold=0.99)
                        
                        curr_coords = pd.Series([rcn, 'highlevel', pr] + coords, index=roi_coords.columns)
                        roi_coords = roi_coords.append(curr_coords, ignore_index=True)
                    else:
                        print(f'File does not exist: {control_zstat_path}')
                else:
                    print(f'File does not exist: {roi_path}')
            
        roi_coords.to_csv(f'{roi_dir}/spheres/sphere_coords.csv', index=False)

# Call the function
#extract_roi_coords()




def extract_roi_sphere(img, coords):
    roi_masker = input_data.NiftiSpheresMasker([tuple(coords)], radius = 6)
    seed_time_series = roi_masker.fit_transform(img)
    
    phys = np.mean(seed_time_series, axis= 1)
    #phys = (phys - np.mean(phys)) / np.std(phys) #TRY WITHOUT STANDARDIZING AT SOME POINT
    phys = phys.reshape((phys.shape[0],1))
    
    return phys

#these paths are tricky
def make_psy_cov(runs,ss):
    rois = ['LO']
    tsk = 'loc'
    rr = 'LO'
    ss = '064'
    runs = [1,2,3]
    
    raw_dir = params.raw_dir
    temp_dir = f'{raw_dir}/sub-{ss}/ses-01' #raw_dir is from hemispace
    cov_dir = f'{temp_dir}/covs'
    
    times = np.arange(0, vols*len(runs), tr)
    full_cov = pd.DataFrame(columns = ['onset','duration', 'value'])
    
    for rn, run in enumerate(runs):    
        
        curr_cov = pd.read_csv(f'{cov_dir}/catloc_{ss}_run-0{run}_Object.txt', sep = '\t', header = None, names = ['onset','duration', 'value'])
        curr_cov_path = f'{cov_dir}/catloc_{ss}_run-0{run}_Object.txt'
        print(f'Loaded curr_cov from: {curr_cov_path}')
        print(curr_cov)
        #contrasting (neg) cov

        curr_cont = pd.read_csv(f'{cov_dir}/catloc_{ss}_run-0{run}_Scramble.txt', sep = '\t', header =None, names =['onset','duration', 'value'])
        curr_cont_path = f'{cov_dir}/catloc_{ss}_run-0{run}_Scramble.txt'
        print(f'Loaded curr_cont from: {curr_cont_path}')
        print(curr_cont)
        curr_cont.iloc[:,2] = curr_cont.iloc[:,2] *-1 #make contrasting cov neg
        
        curr_cov = curr_cov.append(curr_cont) #append to positive

        curr_cov['onset'] = curr_cov['onset'] + (vols*rn)
        full_cov = full_cov.append(curr_cov)
        #add number of vols to the timing cols based on what run you are on
        #e.g., for run 1, add 0, for run 2, add 321
        #curr_cov['onset'] = curr_cov['onset'] + ((rn_n)*vols) 
        
        
        #append to concatenated cov
    full_cov = full_cov.sort_values(by =['onset'])
    cov = full_cov.to_numpy()

    #convolve to hrf
    psy, name = glm.first_level.compute_regressor(cov.T, 'spm', times)
        

    return psy
    
runs = [1,2,3]
subs = ['064']

def conduct_ppi():
    for ss in subs:
        print(ss)
        sub_dir = f'{study_dir}/sub-{ss}/ses-01/' #study is PTOC
        roi_dir = f'{sub_dir}/derivatives/rois' #rois in PTOC
        exp_dir = f'{sub_dir}/derivatives/fsl/{exp}' #PTOC
        
        raw_dir = params.raw_dir
        temp_dir = f'{raw_dir}/sub-{ss}/ses-01' #hemispace 
        cov_dir = f'{temp_dir}/covs' #hemispace

        roi_coords = pd.read_csv(f'{roi_dir}/spheres/sphere_coords.csv') #load ROI coordinates
                                    
        for rr in rois:
            all_runs = [] #this will get filled with the data from each run
            for rcn, rc in enumerate(run_combos): #determine which runs to use for creating ROIs | run combos
                curr_coords = roi_coords[(roi_coords['index'] == rcn) & (roi_coords['task'] =='loc') & (roi_coords['roi'] ==rr)]

                filtered_list = []
                
                for rn in rc:
                    print (rn)
                    run_path = f'{temp_dir}/derivatives/fsl/loc/run-0{rn}/1stLevel.feat/filtered_func_data_reg.nii.gz'
                    #this is where the script is getting stuck
                    if os.path.exists(run_path):
                        curr_run = image.load_img(run_path) #load image data
                        curr_run = image.clean_img(curr_run, standardize=True)
                        filtered_list.append(curr_run)
                        print(f'Loaded {run_path}')
                    else:
                        print(f"File {run_path} does not exist.")
                    
                img4d = image.concat_imgs(filtered_list)
                phys = extract_roi_sphere(img4d,curr_coords[['x','y','z']].values.tolist()[0]) #clarify which coords 
                
                #load behavioral data
                psy = make_psy_cov(rc, ss) #load psy covariates
                
                #combine phys (seed TS) and psy (task TS) into a regressor 
                confounds = pd.DataFrame(columns =['psy', 'phys'])
                confounds['psy'] = psy[:,0]
                confounds['phys'] =phys[:,0]

                #create PPI cov by multiply psy * phys #this is creating the interaction term, the is the PPI time course. There are the individual, so we can get a brain time series with sine phys regressed out
                ppi = psy*phys
                ppi = ppi.reshape((ppi.shape[0],1))

                brain_time_series = brain_masker.fit_transform(img4d, confounds=[confounds]) #change this line to remove confounds 
                brain_time_series_4FC = brain_masker.fit_transform(img4d) #change this line to remove confounds

                #Correlate interaction term to TS for vox in the brain
                seed_to_voxel_correlations = (np.dot(brain_time_series.T, ppi) /
                                ppi.shape[0])
                print(ss, rr, tsk, seed_to_voxel_correlations.max())
                
                #Correlate interaction term to TS for vox in the brain
                seed_to_voxel_correlations = (np.dot(brain_time_series_4FC.T, psy) /
                                psy.shape[0])
                
                seed_to_voxel_correlations = np.arctanh(seed_to_voxel_correlations) # transform back to brain space
                #transform correlation map back to brain
                seed_to_voxel_correlations_img = brain_masker.inverse_transform(seed_to_voxel_correlations.T)
                
                all_runs.append(seed_to_voxel_correlations_img)

            mean_fc = image.mean_img(all_runs)
                
            nib.save(mean_fc, f'{out_dir}/{ss}_{rr}_ppi.nii.gz') #creates the summary file for the PPI analysis (stop here each seed region and the rest of the brain)
            nib.save(mean_fc, f'{out_dir}/{ss}_{rr}_fc_4FC.nii.gz') #creates the summary file for the PSY analysis

conduct_ppi() 



