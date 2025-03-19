# better run as a single script as it takes a while to run
import os
import pandas as pd
import numpy as np
import nibabel as nib
from nilearn.input_data import NiftiSpheresMasker
import sys
import time

# Import parameters
curr_dir = f'/user_data/csimmon2/git_repos/ptoc'
sys.path.insert(0, curr_dir)
import ptoc_params as params

# Set up directories and parameters
study = 'ptoc'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
results_dir = '/user_data/csimmon2/git_repos/ptoc/results'
raw_dir = params.raw_dir

# Load subject information
sub_info = pd.read_csv(f'{curr_dir}/sub_info.csv')
subjects = sub_info[sub_info['group'] == 'control']['sub'].tolist()

# Define the ROIs we're interested in
rois = ['pIPS', 'LO', 'PFS', 'aIPS']
hemispheres = ['left', 'right']
analysis_types = ['fc', 'ppi']  # Process both fc and ppi
task = 'loc'

def create_roi_masker(coords):
    return NiftiSpheresMasker([tuple(coords)], radius=6)

def extract_mean_activation(results_img, roi_masker):
    roi_data = roi_masker.fit_transform(results_img)
    return np.mean(roi_data)

def process_subjects():
    results = []
    processed_count = 0
    
    for subject in subjects:
        print(f"\n=======================================")
        print(f"Processing subject: {subject}")
        print(f"=======================================")
            
        # Load ROI coordinates for all ROIs
        roi_dir = f'{study_dir}/{subject}/ses-01/derivatives/rois'
        roi_coords_file = f'{roi_dir}/spheres/sphere_coords_hemisphere.csv'
        
        if not os.path.exists(roi_coords_file):
            print(f"No coordinates file found for {subject}")
            continue
            
        roi_coords = pd.read_csv(roi_coords_file)
        
        # Create ROI maskers for all ROIs
        roi_maskers = {}
        for roi in rois:
            for hemisphere in hemispheres:
                curr_coords = roi_coords[
                    (roi_coords['task'] == task) & 
                    (roi_coords['roi'] == roi) &
                    (roi_coords['hemisphere'] == hemisphere)
                ]
                
                if curr_coords.empty:
                    print(f"No coordinates found for {roi}, {hemisphere}, {task}")
                    continue
                
                coords = curr_coords[['x', 'y', 'z']].values.tolist()[0]
                roi_maskers[(roi, hemisphere)] = create_roi_masker(coords)
        
        # Process each analysis type
        for curr_analysis in analysis_types:
            print(f"\nProcessing {curr_analysis} analysis for {subject}")
            
            # Check for seed files
            seed_files = []
            for roi in rois:
                for hemisphere in hemispheres:
                    filename = f"{subject}_{roi}_{hemisphere}_{task}_{curr_analysis}.nii.gz"
                    filepath = os.path.join(study_dir, subject, 'ses-01', 'derivatives', 'fc', filename)
                    if os.path.exists(filepath):
                        seed_files.append((roi, hemisphere, filepath))
                    else:
                        print(f"File not found: {filename}")
            
            if not seed_files:
                print(f"No {curr_analysis} seed files found for {subject}")
                continue
                
            print(f"Found {len(seed_files)} {curr_analysis} seed files for {subject}")
            
            # Process each seed file
            for seed_roi, seed_hemisphere, seed_file_path in seed_files:
                # Load the seed results image
                seed_img = nib.load(seed_file_path)
                
                # Extract mean activation in all target ROIs (except the seed ROI)
                for target_roi in rois:
                    # Skip self-connections
                    if target_roi == seed_roi:
                        continue
                        
                    for target_hemisphere in hemispheres:
                        target_masker = roi_maskers.get((target_roi, target_hemisphere))
                        if target_masker is None:
                            continue
                        
                        # Extract mean activation
                        try:
                            mean_activation = extract_mean_activation(seed_img, target_masker)
                            processed_count += 1
                            
                            # Store the results
                            results.append({
                                'subject': subject,
                                'seed_roi': seed_roi,
                                'seed_hemisphere': seed_hemisphere,
                                'target_roi': target_roi,
                                'target_hemisphere': target_hemisphere,
                                'task': task,
                                'analysis': curr_analysis,
                                'mean_activation': mean_activation
                            })
                            
                            print(f"  Extracted: {seed_roi}-{seed_hemisphere} → {target_roi}-{target_hemisphere}: {mean_activation:.4f}")
                                
                        except Exception as e:
                            print(f"Error extracting activation for {seed_roi}-{seed_hemisphere} to {target_roi}-{target_hemisphere}: {e}")
    
    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)
    
    if len(results) == 0:
        print("No results were generated. Check the script output for errors.")
        return None
    
    # Save the results to a CSV file
    output_file = os.path.join(results_dir, 'fc_ppi', 'all_roi_connectivity.csv')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    print(f"Total connections processed: {processed_count}")
    
    return results_df

# Run the processing
if __name__ == "__main__":
    print(f"Starting connectivity extraction for {', '.join(analysis_types)}...")
    start_time = time.time()
    results_df = process_subjects()
    end_time = time.time()
    
    if results_df is not None:
        print(f"Processing completed in {(end_time - start_time)/60:.2f} minutes")
        
        # Print summary statistics
        print("\nSummary of extracted data:")
        print(f"Total subjects: {results_df['subject'].nunique()}")
        print(f"Total connections: {len(results_df)}")
        
        # Count by analysis type
        fc_count = len(results_df[results_df['analysis'] == 'fc'])
        ppi_count = len(results_df[results_df['analysis'] == 'ppi'])
        print(f"FC connections: {fc_count}")
        print(f"PPI connections: {ppi_count}")
        
        # Connections by ROI pair
        print("\nConnections by ROI pair:")
        for seed_roi in rois:
            for target_roi in rois:
                if seed_roi != target_roi:
                    count = len(results_df[(results_df['seed_roi'] == seed_roi) & 
                                          (results_df['target_roi'] == target_roi)])
                    print(f"{seed_roi} → {target_roi}: {count}")
    else:
        print("No results were generated.")