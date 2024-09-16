import os
import pandas as pd
import numpy as np
import nibabel as nib
from nilearn.input_data import NiftiSpheresMasker
import glob
import sys

# Import your parameters
curr_dir = f'/user_data/csimmon2/git_repos/ptoc'
sys.path.insert(0, curr_dir)
import ptoc_params as params

# Set up directories and parameters
study = 'ptoc'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
results_dir = '/user_data/csimmon2/git_repos/ptoc/results'
raw_dir = params.raw_dir

subjects = ['sub-025']  # Add more subjects as needed
rois = ['pIPS', 'LO']  # Update this list as needed
hemispheres = ['left', 'right']
analyses = ['fc', 'ppi']
tasks = ['loc']  # Add more tasks if needed

def extract_mean_activation(results_file, coords):
    # Load the results image
    results_img = nib.load(results_file)
    
    # Create ROI mask
    roi_masker = NiftiSpheresMasker([tuple(coords)], radius=6)
    
    # Extract mean activation within the ROI
    roi_data = roi_masker.fit_transform(results_img)
    mean_activation = np.mean(roi_data)
    
    return mean_activation

def process_subjects():
    results = []
    
    for subject in subjects:
        print(f"Processing subject: {subject}")
        
        # Load ROI coordinates
        roi_dir = f'{study_dir}/{subject}/ses-01/derivatives/rois'
        roi_coords = pd.read_csv(f'{roi_dir}/spheres/sphere_coords_hemisphere.csv')
        
        for roi in rois:
            for hemisphere in hemispheres:
                for analysis in analyses:
                    for task in tasks:
                        # Construct the path to the results file
                        results_dir = os.path.join(study_dir, subject, 'ses-01', 'derivatives', 'fc')
                        results_file = f"{subject}_{roi}_{hemisphere}_{task}_{analysis}.nii.gz"
                        full_results_path = os.path.join(results_dir, results_file)
                        
                        if not os.path.exists(full_results_path):
                            print(f"No {analysis} results file found for {subject}, {roi}, {hemisphere}, {task}")
                            continue
                        
                        # Get coordinates for this ROI and hemisphere
                        curr_coords = roi_coords[
                            (roi_coords['task'] == task) & 
                            (roi_coords['roi'] == roi) &
                            (roi_coords['hemisphere'] == hemisphere)
                        ]
                        
                        if curr_coords.empty:
                            print(f"No coordinates found for {subject}, {roi}, {hemisphere}, {task}")
                            continue
                        
                        coords = curr_coords[['x', 'y', 'z']].values.tolist()[0]
                        
                        # Extract mean activation
                        mean_activation = extract_mean_activation(full_results_path, coords)
                        
                        # Store the results
                        results.append({
                            'subject': subject,
                            'roi': roi,
                            'hemisphere': hemisphere,
                            'task': task,
                            'analysis': analysis,
                            'mean_activation': mean_activation
                        })
    
    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)
    
    # Save the results to a CSV file
    output_file = os.path.join(study_dir, 'roi_mean_activations.csv')
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

# Run the processing
process_subjects()