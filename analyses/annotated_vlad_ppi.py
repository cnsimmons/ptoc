#use this script for PPI analyses because includes run loops ||  RUN for one person and one ROI and then for every ROI and every subject. 
import sys
sys.path.insert(0, '/user_data/vayzenbe/GitHub_Repos/docnet/fmri')
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

warnings.filterwarnings('ignore')

'''exp info'''
subs = list(range(1001,1013))
subs = subs + list(range(2013,2019))

study ='spaceloc'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
out_dir = f'{study_dir}/derivatives/fc'
results_dir = '/user_data/vayzenbe/GitHub_Repos/docnet/results'
exp = 'spaceloc'
rois = ['PPC_spaceloc', 'APC_spaceloc', 'PPC_depthloc', 'APC_depthloc', 'PPC_toolloc', 'APC_toolloc', 'PPC_distloc', 'APC_distloc']
dorsal_rois = ['lPPC', 'lAPC','rPPC', 'rAPC']
dorsal_rois = ['rPPC', 'rAPC']
control_tasks = ['distloc','toolloc','depthloc']
file_suf = '_supp'

'''scan params'''
tr = 1
vols = 321

whole_brain_mask = load_mni152_brain_mask()
mni = load_mni152_template()
brain_masker = input_data.NiftiMasker(whole_brain_mask,
    smoothing_fwhm=0, standardize=True)

'''run info'''
run_num =6
runs = list(range(1,run_num+1))
run_combos = []
#determine the number of left out run combos
for rn1 in range(1,run_num+1):
    for rn2 in range(rn1+1,run_num+1):
        run_combos.append([rn1,rn2])


def extract_roi_coords(): #define seed region in each roi | peak activation in ROI in first run, second run, etc. Saved in sphere coords, this one creates a loop for runs, so update to include the list runs. Creates a list to loop thorugh all possible combinations of runs
    """
    Define ROIs
    """
    parcels = ['PPC', 'APC']

    for ss in subs:
        sub_dir = f'{study_dir}/sub-{study}{ss}/ses-01'
        roi_dir = f'{sub_dir}/derivatives/rois'
        os.makedirs(f'{roi_dir}/spheres', exist_ok=True)
        
        '''make roi spheres for spaceloc'''
        
        exp_dir = f'{sub_dir}/derivatives/fsl'
        parcel_dir = f'{roi_dir}/parcels'
        roi_coords = pd.DataFrame(columns = ['index','task','roi','x','y','z'])
        for rcn, rc in enumerate(run_combos): #determine which runs to use for creating ROIs
            roi_runs = [ele for ele in runs if ele not in rc]

            #load each run
            all_runs = []
            for rn in roi_runs:
                curr_run = image.load_img(f'{exp_dir}/{exp}/run-0{rn}/1stLevel_roi.feat/stats/zstat1_reg.nii.gz')
        
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
                    coords = plotting.find_xyz_cut_coords(mean_zstat,mask_img=roi, activation_threshold = .99) #99% threshold, i.e. highest 1% of voxels

                    masked_stat = image.math_img('img1 * img2', img1=roi, img2=mean_zstat)
                    masked_stat = image.get_data(masked_stat)
                    np_coords = np.where(masked_stat == np.max(masked_stat))
                    #pdb.set_trace()
                    #max_coord = image.coord_transform(np_coords,affine)



                    #masked_image = nib.Nifti1Image(masked_image, affine)  # create the volume image
                    curr_coords = pd.Series([rcn, exp, f'{lr}{pr}'] + coords, index=roi_coords.columns)
                    roi_coords = roi_coords.append(curr_coords,ignore_index = True)

                    '''Create spheres for control task ROIs'''
                    for ct in control_tasks:
                        control_zstat = image.load_img(f'{exp_dir}/{ct}/HighLevel_roi.gfeat/cope1.feat/stats/zstat1.nii.gz')
                        coords = plotting.find_xyz_cut_coords(control_zstat,mask_img=roi, activation_threshold = .99)

                        
                        curr_coords = pd.Series([rcn, ct, f'{lr}{pr}'] + coords, index=roi_coords.columns)
                        roi_coords = roi_coords.append(curr_coords,ignore_index = True) #task, hemisphere, roi, coordinates (i should have access and Vlad no longer does.)

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

def make_psy_cov(runs,ss):
    sub_dir = f'{study_dir}/sub-{study}{ss}/ses-01/'
    cov_dir = f'{sub_dir}/covs'
    times = np.arange(0, vols*len(runs), tr)
    full_cov = pd.DataFrame(columns = ['onset','duration', 'value'])
    
    for rn, run in enumerate(runs):    
        
        curr_cov = pd.read_csv(f'{cov_dir}/SpaceLoc_{study}{ss}_Run{run}_SA.txt', sep = '\t', header = None, names = ['onset','duration', 'value'])
        #contrasting (neg) covariate
        curr_cont = pd.read_csv(f'{cov_dir}/SpaceLoc_{study}{ss}_Run{run}_FT.txt', sep = '\t', header =None, names =['onset','duration', 'value'])
        curr_cont.iloc[:,2] = curr_cont.iloc[:,2] *-1 #make contrasting cov neg
        
        curr_cov = curr_cov.append(curr_cont) #append to positive

        curr_cov['onset'] = curr_cov['onset'] + (vols*rn)
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

#ss = 1001
#rr = 'rPPC_spaceloc'
#pause after this to check the output || run brain_time_seies with out confounds, and instead of ppi you would use phys || change the two lines to be the same, and then you can see the correlation between the two time series. || change the name of the output and the two lines 
def conduct_ppi():
    for ss in subs:
        print(ss)
        sub_dir = f'{study_dir}/sub-{study}{ss}/ses-01/'
        cov_dir = f'{sub_dir}/covs'
        roi_dir = f'{sub_dir}/derivatives/rois'
        exp_dir = f'{sub_dir}/derivatives/fsl/{exp}'

        roi_coords = pd.read_csv(f'{roi_dir}/spheres/sphere_coords.csv')

        for tsk in ['spaceloc','distloc']: #just include one task for now get rid of the loop and extract current coords
            for rr in dorsal_rois:
                all_runs = [] #this will get filled with the data from each run
                for rcn, rc in enumerate(run_combos): #determine which runs to use for creating ROIs | run combos
                    curr_coords = roi_coords[(roi_coords['index'] == rcn) & (roi_coords['task'] ==tsk) & (roi_coords['roi'] ==rr)]

                    filtered_list = []
                    for rn in rc:
                        
                        curr_run = image.load_img(f'{exp_dir}/run-0{rn}/1stLevel.feat/filtered_func_data_reg.nii.gz') #filtered_func data - cope and zstat is a mean image, while the filtered is preprocessed data - standardized
                        #double check the above exists.
                        curr_run = image.clean_img(curr_run,standardize=True)
                        filtered_list.append(curr_run)
                        
                    img4d = image.concat_imgs(filtered_list)
                    phys = extract_roi_sphere(img4d,curr_coords[['x','y','z']].values.tolist()[0]) #extract ROI spehere coordinate, pulls out just the time series from that part of the brain, every voxel of the brain, time series for just the spheres we've pulled out
                    #load behavioral data
                    #CONVOLE TO HRF
                    psy = make_psy_cov(rc, ss) #this is the one that goes to the covariate folder and grabs the covariate we care about and converts the three coloumn into binary data. 

                    #combine phys (seed TS) and psy (task TS) into a regressor ||  TS = time series, CNS
                    confounds = pd.DataFrame(columns =['psy', 'phys'])
                    confounds['psy'] = psy[:,0]
                    confounds['phys'] =phys[:,0]

                    #create PPI cov by multiply psy * phys #this is createing the interaction term, the is the PPI time course. There are the individual, so we can get a brain time series with sine phys regressed out
                    ppi = psy*phys
                    ppi = ppi.reshape((ppi.shape[0],1))

                    brain_time_series = brain_masker.fit_transform(img4d, confounds=[confounds]) #change this line to remove confounds 
                    #brain_time_series_4FC = brain_masker.fit_transform(img4d) #change this line to remove confounds

                    #Correlate interaction term to TS for vox in the brain
                    seed_to_voxel_correlations = (np.dot(brain_time_series.T, ppi) /
                                    ppi.shape[0])
                    print(ss, rr, tsk, seed_to_voxel_correlations.max())
                    
                    #Correlate interaction term to TS for vox in the brain
                    #seed_to_voxel_correlations = (np.dot(brain_time_series_4FC.T, psy) /
                                    #psy.shape[0])
                    
                    seed_to_voxel_correlations = np.arctanh(seed_to_voxel_correlations) # transform back to brain space
                    #transform correlation map back to brain
                    seed_to_voxel_correlations_img = brain_masker.inverse_transform(seed_to_voxel_correlations.T)
                    
                    all_runs.append(seed_to_voxel_correlations_img)

                mean_fc = image.mean_img(all_runs)
                    
                nib.save(mean_fc, f'{out_dir}/sub-{study}{ss}_{rr}_{tsk}_ppi.nii.gz') #creates the summary file for the PPI analysis (stop here each seed region and the rest of the brain)
                #nib.save(mean_fc, f'{out_dir}/sub-{study}{ss}_{rr}_{tsk}_fc_4FC.nii.gz') #creates the summary file for the PSY analysis


def create_summary(): #strength of the connectivity - to look at later, pause with above. 
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
                




#                    print(ss, roi)

#subs = list(range(2018,2015,-1))
#print(subs)
#extract_roi_coords()
#conduct_ppi()
subs = list(range(1001,1013)) + list(range(2013,2019))
create_summary()
#make_psy_cov(1001,[1,2])










        







