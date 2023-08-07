import sys
curr_dir = '/user_data/csimmon2/git_repos/ptoc/fmri'
sys.path.append(curr_dir)
#sys.path.insert(0, '/user_data/vayzenbe/GitHub_Repos/docnet/fmri') #change sys/paths
import pandas as pd
from nilearn import image, plotting, input_data, glm
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

import ptoc_params as params #change to study params

warnings.filterwarnings('ignore')

'''exp info'''
sub_info = params.sub_info

study ='ptoc' #change obvs

data_dir = params.data_dir
raw_dir = params.raw_dir  
results_dir = params.results_dir
study_dir = f"/lab_data/behrmannlab/vlad/{study}" 
exp = 'loc' #not sure

file_suf = ''

'''scan params''' #update with params
tr = params.tr
vols = params.vols
cope = params.cope
rois = params.rois

whole_brain_mask = load_mni152_brain_mask()
mni = load_mni152_template()
brain_masker = input_data.NiftiMasker(whole_brain_mask,
    smoothing_fwhm=0, standardize=True)

'''run info'''
run_num =3
runs = list(range(1,run_num+1))
run_combos = []
#determine the number of left out run combos
for rn1 in range(1,run_num+1):
    for rn2 in range(rn1+1,run_num+1):
        run_combos.append([rn1,rn2])


def extract_roi_coords():
    """
    Define ROIs
    """
    
    parcels = ['LO', 'PFS', 'pIPS','aIPS']

    for ss in subs:
        sub_dir = f'{study_dir}/sub-{study}{ss}/ses-01'
        roi_dir = f'{sub_dir}/derivatives/rois'
        os.makedirs(f'{roi_dir}/spheres', exist_ok=True)
        
        '''make roi spheres for spaceloc'''
        
        exp_dir = f'{sub_dir}/derivatives/fsl'
        parcel_dir = f'{roi_dir}/parcels'
        roi_coords = pd.DataFrame(columns = ['index','task','roi','x','y','z']) #create empty df to store roi coords
        for rcn, rc in enumerate(run_combos): #determine which runs to use for creating ROIs


            #load each run
            all_runs = []
            for rn in rc:
                curr_run = image.load_img(f'{exp_dir}/{exp}/run-0{rn}/1stLevel_roi.feat/stats/zstat{cope}_reg.nii.gz')
        
                all_runs.append(curr_run)

            mean_zstat = image.mean_img(all_runs)
            affine = mean_zstat.affine
            
            #loop through parcel determine coord of peak voxel
            for lr in ['l','r']:
                for pr in parcels:

                    #load parcel
                    roi = image.load_img(f'{parcel_dir}/{lr}{pr}.nii.gz')
                    roi = image.math_img('img > 0', img=roi)

                    #masked_image = roi*image.get_data(mean_zstat)
                    coords = plotting.find_xyz_cut_coords(mean_zstat,mask_img=roi, activation_threshold = .99) #top 1% of voxels

                    masked_stat = image.math_img('img1 * img2', img1=roi, img2=mean_zstat)
                    masked_stat = image.get_data(masked_stat)
                    np_coords = np.where(masked_stat == np.max(masked_stat))
                    #pdb.set_trace()
                    #max_coord = image.coord_transform(np_coords,affine)



                    #masked_image = nib.Nifti1Image(masked_image, affine)  # create the volume image
                    curr_coords = pd.Series([rcn, exp, f'{lr}{pr}'] + coords, index=roi_coords.columns)
                    roi_coords = roi_coords.append(curr_coords,ignore_index = True)


        roi_coords.to_csv(f'{roi_dir}/spheres/sphere_coords.csv', index=False) #saves coords to csv | where our most object selective regions/voxels are

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

def conduct_fc():

    #for each subject, load their individual anatomy and mask

    for sub, group, hemi in zip(sub_info['sub'], sub_info['group'], sub_info['intact_hemi']):
        print(sub)
        sub_dir = f'{study_dir}/sub-{study}{ss}/ses-01/'
        cov_dir = f'{sub_dir}/covs'
        roi_dir = f'{sub_dir}/derivatives/rois'
        exp_dir = f'{sub_dir}/derivatives/fsl/{exp}'
        
        if hemi == 'both':
            hemis = ['l','r']
        elif hemi == 'left':
            hemis = ['l']
        elif hemi == 'right':
            hemis = ['r']
        

        roi_coords = pd.read_csv(f'{roi_dir}/spheres/sphere_coords.csv')

        for tsk in ['loc']: #change to just loc
            for lr in hemis:
                if lr == 'l':
                    curr_hemi = 'left'
                elif lr == 'r':
                    curr_hemi = 'right'
                #load anat mask
                anat_mask = image.load_img(f'{params.raw_dir}/{sub}/ses-01/anat/sub-{study}{sub}_ses-01_T1w_brain_mask_{curr_hemi}.nii.gz')
                #binarize mask
                anat_mask = image.math_img('img > 0', img=anat_mask)
                brain_masker = input_data.NiftiMasker(anat_mask, smoothing_fwhm=0, standardize=True)
                
                
                for rr in rois: #change to just rois
                    roi = f'{lr}{rr}'
                    all_runs = [] #this will get filled with the data from each run
                    for rcn, rc in enumerate(run_combos): #determine which runs to use for creating ROIs
                        curr_coords = roi_coords[(roi_coords['index'] == rcn) & (roi_coords['task'] ==tsk) & (roi_coords['roi'] ==roi)]

                        #Determine which run was not used to create the ROI
                        test_runs = [elem for elem in runs if elem not in rc ]

                        filtered_list = []
                        for rn in test_runs:
                            #extract roi time series dat afrom held out run
                            curr_run = image.load_img(f'{exp_dir}/run-0{rn}/1stLevel.feat/filtered_func_data_reg.nii.gz')
                            curr_run = image.clean_img(curr_run,standardize=True)
                            filtered_list.append(curr_run)
                            
                            
                        img4d = image.concat_imgs(filtered_list)
                        #extract time series from peak voxel
                        phys = extract_roi_sphere(img4d,curr_coords[['x','y','z']].values.tolist()[0])

                        #Extract time series from rest of brain
                        brain_time_series = brain_masker.fit_transform(img4d)

                        #Correlate interaction term to TS for vox in the brain
                        seed_to_voxel_correlations = (np.dot(brain_time_series.T, phys) /
                                        phys.shape[0])
                        print(sub, roi, tsk, seed_to_voxel_correlations.max())
                        
                        #convert correlations to fisher z
                        seed_to_voxel_correlations = np.arctanh(seed_to_voxel_correlations)

                        #transform correlation map back to brain space
                        seed_to_voxel_correlations_img = brain_masker.inverse_transform(seed_to_voxel_correlations.T)
                        
                        all_runs.append(seed_to_voxel_correlations_img)

                    mean_fc = image.mean_img(all_runs)
                        
                    nib.save(mean_fc, f'{results_dir}/sub-{study}{sub}_{roi}_{tsk}_fc.nii.gz')

extract_roi_coords() #temp location

conduct_fc() #temp location

quit()

def create_summary():
    """
    extract avg FC from target ROIs

    CLAIRE IGNORE THIS FUNCITON FOR NOW
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
            
            for ss in subs:
                
                sub_dir = f'{study_dir}/sub-{study}{ss}/ses-01/'
                roi_dir = f'{sub_dir}/derivatives/rois'
                
                #if os.path.exists(f'{roi_dir}/{ventral}_peak.nii.gz'):
                ventral_mask = image.load_img(f'{roi_dir}/{ventral}.nii.gz')
                ventral_mask = input_data.NiftiMasker(ventral_mask)
                
                
                roi_mean = []
                roi_mean.append(ss)
                
                #For each dorsal ROI
                for lr in ['l','r']:
                    for rr in rois:
                        
                        roi = f'{lr}{rr}'
                        if os.path.exists(f'{out_dir}/sub-{study}{ss}_{roi}_fc.nii.gz'):
                            ppi_img = image.load_img(f'{out_dir}/sub-{study}{ss}_{roi}_fc.nii.gz')
                            #ppi_img  = image.smooth_img(ppi_img, 6)
                            acts = ventral_mask.fit_transform(ppi_img)

                            
                            roi_mean.append(acts.mean())
                        else:
                            roi_mean.append(np.nan)
            #pdb.set_trace()
                summary_df = summary_df.append(pd.Series(roi_mean, index = summary_df.columns), ignore_index = True)
        #print(ventral)
            summary_df.to_csv(f'{results_dir}/ppi/{ventral}_fc{file_suf}.csv', index=False)
        #summary_df.iloc[:, 1:].mean().plot.bar()
        #plt.pause(0.0001)
                





extract_roi_coords()

conduct_fc()









        







