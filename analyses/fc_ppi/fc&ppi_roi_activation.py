import os
import pandas as pd
import numpy as np
import nibabel as nib
from nilearn.input_data import NiftiSpheresMasker
from nilearn import image
import sys
import time

# Import your parameters
curr_dir = f'/user_data/csimmon2/git_repos/ptoc'
sys.path.insert(0, curr_dir)
import ptoc_params as params

# Set up directories and parameters
study = 'ptoc'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
results_dir = '/user_data/csimmon2/git_repos/ptoc/results'
raw_dir = params.raw_dir

#subjects = ['sub-025']  # Add more subjects as needed
sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
sub_info = sub_info[sub_info['group'] == 'control']
subjects = sub_info['sub'].tolist()
#subjects = ['sub-025']

rois = ['pIPS', 'LO']  # Update this list as needed
hemispheres = ['left', 'right']
analyses = ['fc', 'ppi']
tasks = ['loc']  # Add more tasks if needed

def create_roi_masker(coords):
    start_time = time.time()
    roi_masker = NiftiSpheresMasker([tuple(coords)], radius=6)
    end_time = time.time()
    print(f"Time to create ROI masker: {end_time - start_time:.2f} seconds")
    return roi_masker

def extract_mean_activation(results_img, roi_masker):
    start_time = time.time()
    roi_data = roi_masker.fit_transform(results_img)
    mean_activation = np.mean(roi_data)
    end_time = time.time()
    print(f"Time to extract mean activation: {end_time - start_time:.2f} seconds")
    return mean_activation

def process_subjects():
    overall_start_time = time.time()
    results = []
    
    for subject in subjects:
        subject_start_time = time.time()
        print(f"Processing subject: {subject}")
        
        # Load ROI coordinates
        roi_dir = f'{study_dir}/{subject}/ses-01/derivatives/rois'
        roi_coords = pd.read_csv(f'{roi_dir}/spheres/sphere_coords_hemisphere.csv')
        
        # Create ROI maskers for each ROI/hemisphere combination
        roi_maskers = {}
        for roi in rois:
            for hemisphere in hemispheres:
                for task in tasks:
                    curr_coords = roi_coords[
                        (roi_coords['task'] == task) & 
                        (roi_coords['roi'] == roi) &
                        (roi_coords['hemisphere'] == hemisphere)
                    ]
                    
                    if curr_coords.empty:
                        print(f"No coordinates found for {subject}, {roi}, {hemisphere}, {task}")
                        continue
                    
                    coords = curr_coords[['x', 'y', 'z']].values.tolist()[0]
                    roi_maskers[(roi, hemisphere, task)] = create_roi_masker(coords)
        
        # Process each analysis
        for analysis in analyses:
            analysis_start_time = time.time()
            results_dir = os.path.join(study_dir, subject, 'ses-01', 'derivatives', 'fc')
            
            for roi in rois:
                for hemisphere in hemispheres:
                    for task in tasks:
                        results_file = f"{subject}_{roi}_{hemisphere}_{task}_{analysis}.nii.gz"
                        full_results_path = os.path.join(results_dir, results_file)
                        
                        if not os.path.exists(full_results_path):
                            print(f"No {analysis} results file found for {subject}, {roi}, {hemisphere}, {task}")
                            continue
                        
                        # Load the results image
                        load_start_time = time.time()
                        results_img = nib.load(full_results_path)
                        load_end_time = time.time()
                        print(f"Time to load results image: {load_end_time - load_start_time:.2f} seconds")
                        
                        # Get the corresponding ROI masker
                        roi_masker = roi_maskers.get((roi, hemisphere, task))
                        if roi_masker is None:
                            print(f"No ROI masker found for {subject}, {roi}, {hemisphere}, {task}")
                            continue
                        
                        # Extract mean activation
                        mean_activation = extract_mean_activation(results_img, roi_masker)
                        
                        # Store the results
                        results.append({
                            'subject': subject,
                            'roi': roi,
                            'hemisphere': hemisphere,
                            'task': task,
                            'analysis': analysis,
                            'mean_activation': mean_activation
                        })
            
            analysis_end_time = time.time()
            print(f"Time to process {analysis} analysis: {analysis_end_time - analysis_start_time:.2f} seconds")
        
        subject_end_time = time.time()
        print(f"Total time to process subject {subject}: {subject_end_time - subject_start_time:.2f} seconds")
    
    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)
    
    # Save the results to a CSV file
    output_file = os.path.join(results_dir, 'roi_mean_activations.csv')
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    overall_end_time = time.time()
    print(f"Total execution time: {overall_end_time - overall_start_time:.2f} seconds")

# Run the processing
process_subjects()