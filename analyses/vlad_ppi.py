import sys
curr_dir = '/user_data/vayzenbe/GitHub_Repos/bwoc'
sys.path.insert(0, curr_dir)
import pandas as pd
from nilearn import image, plotting, input_data, glm, maskers
#from nilearn.glm import threshold_stats_img
import numpy as np

from nilearn.input_data import NiftiMasker
import nibabel as nib


import os
import statsmodels.api as sm
from nilearn.datasets import load_mni152_brain_mask, load_mni152_template
import matplotlib.pyplot as plt
import pdb
from scipy.stats import gamma
import warnings

import bwoc_params as params

warnings.filterwarnings('ignore')

'''exp info'''

subs = params.sub_info['loc_sub'].tolist()


out_dir = f'{params.scratch_dir}/derivatives/ppi'
os.makedirs(out_dir, exist_ok=True)

cov_dir = f'{params.loc_data}'
results_dir = f'/{curr_dir}/results'
roi_suf = 'toolloc'
rois = ['PPC', 'APC', 'LO', 'PFS']

first_fix = 6

runs = [1,2]

file_suf = ''



whole_brain_mask = '/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'
mni = '/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz'
brain_masker = input_data.NiftiMasker(whole_brain_mask,
    smoothing_fwhm=0, standardize=True)


def extract_roi_coords(parcels):
    """
    Define ROIs
    """
    

    for sub in subs:
        sub_dir = f'{params.study_dir}/sub-{sub}/ses-01'
        roi_dir = f'{sub_dir}/derivatives/rois'
        os.makedirs(f'{roi_dir}/spheres', exist_ok=True)
        
        '''make roi spheres for spaceloc'''
        
        exp_dir = f'{sub_dir}/derivatives/fsl'
        parcel_dir = f'{roi_dir}/parcels'
        roi_coords = pd.DataFrame(columns = ['task','roi','x','y','z'])
        
        
        zstat = image.load_img(f'{exp_dir}/{params.task}/HighLevel.gfeat/cope{params.cope}.feat/stats/zstat1.nii.gz')
        affine = zstat.affine
        
        #loop through parcel determine coord of peak voxel
        for lr in ['l','r']:
            for pr in parcels:

                #load parcel
                roi = image.load_img(f'{parcel_dir}/{lr}{pr}.nii.gz')
                roi = image.binarize_img(roi, threshold = .5)

                
                masker = maskers.NiftiMasker(mask_img=roi)
                masker.fit(zstat)
                roi_data = masker.transform(zstat)

                
                coords = plotting.find_xyz_cut_coords(zstat,mask_img=roi, activation_threshold = .99)

                
                #masked_stat = image.get_data(roi_data)
                #np_coords = np.where(roi_data == np.max(roi_data))

                

                #masked_image = nib.Nifti1Image(masked_image, affine)  # create the volume image
                curr_coords = pd.Series([params.task, f'{lr}{pr}'] + coords, index=roi_coords.columns)
                roi_coords = roi_coords.append(curr_coords,ignore_index = True)

        roi_coords.to_csv(f'{roi_dir}/spheres/sphere_coords.csv', index=False)

def extract_roi_sphere(img, coords):
    roi_masker = input_data.NiftiSpheresMasker([tuple(coords)], radius = 6)
    seed_time_series = roi_masker.fit_transform(img)
    
    phys = np.mean(seed_time_series, axis= 1)
    #phys = (phys - np.mean(phys)) / np.std(phys) #TRY WITHOUT STANDARDIZING AT SOME POINT
    phys = phys.reshape((phys.shape[0],1))
    
    return phys

"""
def load_filtered_func(run):
    curr_img = image.load_img(f'{exp_dir}/run-0{run}/1stLevel.feat/filtered_func_data_reg.nii.gz')
    #curr_img = image.clean_img(curr_img,standardize=True, t_r=1)
    
    img4d = image.resample_to_img(curr_img,mni)
    
    roi_masker = input_data.NiftiMasker(roi_mask)
    seed_time_series = roi_masker.fit_transform(img4d)
    
    phys = np.mean(seed_time_series, axis= 1)
    phys = (phys - np.mean(phys)) / np.std(phys)
    phys = phys.reshape((phys.shape[0],1))
    
    return img4d, phys
"""    

def make_psy_cov(runs,sub):
    sub_dir = f'{params.study_dir}/sub-{sub}/ses-01/'
    cov_dir = f'{params.loc_data}/sub-{sub}/ses-01/covs'
    times = np.arange(0, params.vols*len(runs), params.tr)
    full_cov = pd.DataFrame(columns = ['onset','duration', 'value'])
    for rn, run in enumerate(runs):    
        #load the two tool loc covs
        pos_cov1 = pd.read_csv(f'{cov_dir}/ToolLoc_{sub}_run{run}_tool.txt', sep = '\t', header = None, names = ['onset','duration', 'value'])
        pos_cov2 = pd.read_csv(f'{cov_dir}/ToolLoc_{sub}_run{run}_non_tool.txt', sep = '\t', header = None, names = ['onset','duration', 'value'])

        #load contrasting (neg) cov
        curr_cont = pd.read_csv(f'{cov_dir}/ToolLoc_{sub}_run{run}_scramble.txt', sep = '\t', header = None, names = ['onset','duration', 'value'])
        curr_cont.iloc[:,2] = curr_cont.iloc[:,2] *-1 #make contrasting cov neg
        
        curr_cov = pos_cov1.append(pos_cov2) #append the positive covs
        curr_cov = curr_cov.append(curr_cont) #append to positive

        curr_cov['onset'] = curr_cov['onset'] + (params.vols*rn)
        full_cov = full_cov.append(curr_cov)
        #add number of vols to the timing cols based on what run you are on
        #e.g., for run 1, add 0, for run 2, add 321
        #curr_cov['onset'] = curr_cov['onset'] + ((rn_n)*vols) 
        #pdb.set_trace()
        
        #append to concatenated cov
    full_cov = full_cov.sort_values(by =['onset'])
    cov = full_cov.to_numpy()

    #convolve to hrf
    psy, name = glm.first_level.compute_regressor(cov.T, 'spm', times)
        

    return psy
#runs = [1]


def conduct_ppi():
    for sub in subs:
        print(sub)
        sub_dir = f'{params.study_dir}/sub-{sub}/ses-01/'
        cov_dir = f'{sub_dir}/covs'
        roi_dir = f'{sub_dir}/derivatives/rois'
        exp_dir = f'{sub_dir}/derivatives/fsl/{params.task}'

        roi_coords = pd.read_csv(f'{roi_dir}/spheres/sphere_coords.csv')

        for tsk in ['toolloc']:
            for lr in ['l','r']:
                    
                for roi in rois:
                
                
                    curr_coords = roi_coords[(roi_coords['task'] ==tsk) & (roi_coords['roi'] ==f'{lr}{roi}')]

                    filtered_list = []
                    for run in runs:
                        
                        curr_run = image.load_img(f'{exp_dir}/run-0{run}/1stLevel.feat/filtered_func_data_reg.nii.gz')
                        affine = curr_run.affine
                        
                        curr_run = image.clean_img(curr_run,standardize=True)
                        filtered_list.append(curr_run)
                        
                    img4d = image.concat_imgs(filtered_list)
                    
                    phys = extract_roi_sphere(img4d,curr_coords[['x','y','z']].values.tolist()[0])
                    #load behavioral data
                    #CONVOLE TO HRF
                    psy = make_psy_cov(runs, sub)

                    #combine phys (seed TS) and psy (task TS) into a regresubor
                    confounds = pd.DataFrame(columns =['psy', 'phys'])
                    confounds['psy'] = psy[:,0]
                    confounds['phys'] =phys[:,0]

                    #create PPI cov by multiply psy * phys
                    ppi = psy*phys
                    ppi = ppi.reshape((ppi.shape[0],1))

                    brain_time_series = brain_masker.fit_transform(img4d, confounds=[confounds])

                    #Correlate interaction term to TS for vox in the brain
                    seed_to_voxel_correlations = (np.dot(brain_time_series.T, ppi) /
                                    ppi.shape[0])
                    print(sub, roi, tsk, seed_to_voxel_correlations.max())
                    
                    seed_to_voxel_correlations = np.arctanh(seed_to_voxel_correlations)
                    #transform correlation map back to brain
                    seed_to_voxel_correlations_img = brain_masker.inverse_transform(seed_to_voxel_correlations.T)
                    
                    

                    mean_fc = seed_to_voxel_correlations_img
                        
                    nib.save(mean_fc, f'{out_dir}/sub-{sub}_{lr}{roi}_{tsk}_fc.nii.gz')


def create_summary():
    """
    extract avg PPI in LO  and PFS
    """
    ventral_rois = ['LO_toolloc']
    #rois = ["PPC_spaceloc", "PPC_distloc", "PPC_toolloc"]
    rois = ["PPC_spaceloc", "APC_spaceloc", "APC_distloc", "APC_toolloc"]
    print(subs)
    #For each ventral ROI
    for lrv in ['l','r']:
        
        for vr in ventral_rois:
            
            summary_df = pd.DataFrame(columns = ['sub'] + ['l' + rr for rr in rois] + ['r' + rr for rr in rois])
            #summary_df = pd.DataFrame(columns = ['sub'] + ['r' + rr for rr in rois])
            ventral = f'{lrv}{vr}'
            print(ventral)
            
            for sub in subs:
                
                sub_dir = f'{params.study_dir}/sub-{sub}/ses-01/'
                roi_dir = f'{sub_dir}/derivatives/rois'
                
                #if os.path.exists(f'{roi_dir}/{ventral}_peak.nii.gz'):
                ventral_mask = image.load_img(f'{roi_dir}/{ventral}.nii.gz')
                ventral_mask = input_data.NiftiMasker(ventral_mask)
                
                
                roi_mean = []
                roi_mean.append(sub)
                
                #For each dorsal ROI
                for lr in ['l','r']:
                    for rr in rois:
                        
                        roi = f'{lr}{rr}'
                        if os.path.exists(f'{out_dir}/sub-{sub}_{roi}_fc.nii.gz'):
                            ppi_img = image.load_img(f'{out_dir}/sub-{sub}_{roi}_fc.nii.gz')
                            #ppi_img  = image.smooth_img(ppi_img, 6)
                            acts = ventral_mask.fit_transform(ppi_img)

                            
                            roi_mean.append(acts.mean())
                        else:
                            roi_mean.append(np.nan)

                summary_df = summary_df.append(pd.Series(roi_mean, index = summary_df.columns), ignore_index = True)

            summary_df.to_csv(f'{results_dir}/ppi/{ventral}_fc{file_suf}.csv', index=False)



#extract_roi_coords(rois)


conduct_ppi()







        