#!/usr/bin/env python
"""
Script to create connectivity matrices using the integrated parcellation
of specialized ROIs (LO, pIPS, aIPS, PFS) and Glasser atlas regions.

This version focuses only on the Object condition for efficiency.
"""

import os
import sys
import pandas as pd
import numpy as np
import nibabel as nib
from nilearn import image
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn.glm.first_level import compute_regressor
import logging
import argparse
from datetime import datetime

# Set up logging
log_filename = f"integrated_connectivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Create connectivity matrices using integrated parcellation')
    parser.add_argument('--study_dir', type=str, default='/lab_data/behrmannlab/vlad/ptoc',
                        help='Path to study directory')
    parser.add_argument('--code_dir', type=str, default='/user_data/csimmon2/git_repos/ptoc',
                        help='Path to code directory')
    parser.add_argument('--parcellation', type=str, 
                        default='/user_data/csimmon2/git_repos/ptoc/results/integrated_parcellation/integrated_parcellation.nii.gz',
                        help='Path to integrated parcellation file')
    parser.add_argument('--labels', type=str, 
                        default='/user_data/csimmon2/git_repos/ptoc/results/integrated_parcellation/integrated_labels.csv',
                        help='Path to integrated labels file')
    parser.add_argument('--output_dir', type=str, 
                        default='/user_data/csimmon2/git_repos/ptoc/results/integrated_connectivity_object',
                        help='Path to output directory')
    parser.add_argument('--subjects', type=str, default='',
                        help='Comma-separated list of subjects to process (default: all from sub_info.csv)')
    parser.add_argument('--group', type=str, default='control',
                        help='Subject group to process')
    return parser.parse_args()

def setup_environment(args):
    """Set up environment and paths"""
    # Import parameters from project
    sys.path.insert(0, args.code_dir)
    try:
        import ptoc_params as params
        raw_dir = params.raw_dir
    except ImportError:
        logging.warning("Could not import ptoc_params.py, using default raw_dir")
        raw_dir = f"{args.study_dir}/raw"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load subject information
    sub_info_path = f'{args.code_dir}/sub_info.csv'
    if os.path.exists(sub_info_path):
        sub_info = pd.read_csv(sub_info_path)
        logging.info(f"Loaded subject info: {len(sub_info)} subjects")
    else:
        logging.error(f"Subject info file not found: {sub_info_path}")
        sys.exit(1)
    
    # Get subjects to process
    subjects_to_skip = ['sub-084']  # Known problematic subjects
    
    if args.subjects:
        # Use provided subjects
        subs = [s.strip() for s in args.subjects.split(',')]
        logging.info(f"Using {len(subs)} provided subjects")
    else:
        # Get subjects from sub_info
        subs = sub_info[(sub_info['group'] == args.group) & 
                        (~sub_info['sub'].isin(subjects_to_skip))]['sub'].tolist()
        logging.info(f"Found {len(subs)} {args.group} subjects in sub_info")
    
    # Verify integrated parcellation exists
    if not os.path.exists(args.parcellation):
        logging.error(f"Integrated parcellation not found: {args.parcellation}")
        sys.exit(1)
    
    # Verify labels file exists
    if not os.path.exists(args.labels):
        logging.error(f"Integrated labels file not found: {args.labels}")
        sys.exit(1)
    
    return subs, raw_dir

def load_integrated_parcellation(parcellation_path, labels_path):
    """Load integrated parcellation and labels"""
    try:
        # Load parcellation image
        parcellation_img = nib.load(parcellation_path)
        logging.info(f"Loaded integrated parcellation: {parcellation_img.shape}")
        
        # Load labels
        labels_df = pd.read_csv(labels_path)
        logging.info(f"Loaded {len(labels_df)} region labels")
        
        # Count specialized ROIs
        roi_count = len(labels_df[labels_df['type'] == 'specialized_roi'])
        glasser_count = len(labels_df[labels_df['type'] == 'glasser_atlas'])
        logging.info(f"Parcellation contains {roi_count} specialized ROIs and {glasser_count} Glasser regions")
        
        # Verify parcellation doesn't have overlapping regions
        parc_data = parcellation_img.get_fdata()
        unique_values = np.unique(parc_data)
        unique_values = unique_values[unique_values > 0]  # Remove background
        logging.info(f"Parcellation has {len(unique_values)} unique values (excluding background)")
        
        # Check labels match parcellation values
        label_ids = labels_df['id'].unique()
        missing_labels = [v for v in unique_values if v not in label_ids]
        if missing_labels:
            logging.warning(f"Found {len(missing_labels)} regions in parcellation with no label")
        
        # Check if we need to resample the parcellation
        if parcellation_img.shape != (91, 109, 91):
            logging.warning("Parcellation not in standard space, will need to resample")
        
        return parcellation_img, labels_df
        
    except Exception as e:
        logging.error(f"Error loading parcellation: {str(e)}")
        return None, None

def verify_standard_space(img):
    """Verify image is in 2mm standard space"""
    if img.shape[:3] != (91, 109, 91):
        logging.warning(f"Unexpected shape: {img.shape}")
        return False
    
    vox_size = np.sqrt(np.sum(img.affine[:3, :3] ** 2, axis=0))
    if not np.allclose(vox_size, [2., 2., 2.], atol=0.1):
        logging.warning(f"Unexpected voxel size: {vox_size}")
        return False
    
    return True

def get_condition_mask(run_num, ss, n_timepoints, raw_dir):
    """Create a binary mask for timepoints during the Object condition"""
    condition = "Object"  # Focusing only on Object condition
    cov_dir = f'{raw_dir}/{ss}/ses-01/covs'
    ss_num = ss.split('-')[1]
    
    # Load condition timing file
    cov_file = f'{cov_dir}/catloc_{ss_num}_run-0{run_num}_{condition}.txt'
    if not os.path.exists(cov_file):
        logging.warning(f'Covariate file not found: {cov_file}')
        return np.zeros(n_timepoints, dtype=bool)
    
    # Load timing data
    cov = pd.read_csv(cov_file, sep='\t', header=None, 
                      names=['onset', 'duration', 'value'])
    
    # Create timepoints array
    tr = 2.0  # TR in seconds
    times = np.arange(0, n_timepoints * tr, tr)
    
    # Convert timing to binary mask
    condition_reg, _ = compute_regressor(cov.to_numpy().T, 'spm', times)
    
    # Convert to binary mask and ensure it's 1D
    return (condition_reg > 0).ravel()

def create_connectivity_matrix(ss, parcellation_img, raw_dir):
    """Create connectivity matrix for the Object condition using integrated parcellation"""
    logging.info(f"Processing subject {ss} for Object condition")
    
    # Define runs
    run_num = 3
    runs = list(range(1, run_num + 1))
    all_runs_data = []
    
    # Ensure parcellation is in standard space
    parc_is_standard = verify_standard_space(parcellation_img)
    
    for rn in runs:
        # Load standard space data
        run_path = f'{raw_dir}/{ss}/ses-01/derivatives/reg_standard/filtered_func_run-0{rn}_standard.nii.gz'
        
        if not os.path.exists(run_path):
            logging.warning(f'Standard space data not found: {run_path}')
            continue
        
        subject_img = nib.load(run_path)
        
        # Verify standard space
        if not verify_standard_space(subject_img):
            logging.warning(f"Data not in expected standard space for {ss} run-{rn}")
            continue
        
        # Resample parcellation to match subject data if needed
        if not parc_is_standard:
            logging.info("Resampling parcellation to match subject data")
            parcellation_resampled = image.resample_to_img(
                parcellation_img, 
                subject_img, 
                interpolation='nearest'
            )
        else:
            parcellation_resampled = parcellation_img
        
        # Extract time series
        masker = NiftiLabelsMasker(
            labels_img=parcellation_resampled,
            standardize='zscore_sample',
            memory=None,
            verbose=0
        )
        
        time_series = masker.fit_transform(subject_img)
        logging.info(f"Time series shape before masking: {time_series.shape}")
        
        # Get condition mask for Object condition
        condition_mask = get_condition_mask(rn, ss, time_series.shape[0], raw_dir)
        logging.info(f"Condition mask shape: {condition_mask.shape}")
        
        # Only keep timepoints during Object condition
        masked_time_series = time_series[condition_mask]
        logging.info(f"Time series shape after masking: {masked_time_series.shape}")
        
        if masked_time_series.shape[0] > 0:  # Only append if we have data
            all_runs_data.append(masked_time_series)
    
    if not all_runs_data:
        logging.warning(f'No valid data found for subject {ss} Object condition')
        return None
    
    # Concatenate runs
    full_time_series = np.concatenate(all_runs_data, axis=0)
    logging.info(f"Full time series shape: {full_time_series.shape}")
    
    # Compute connectivity matrix
    correlation_measure = ConnectivityMeasure(
        kind='correlation',
        standardize='zscore_sample'
    )
    connectivity_matrix = correlation_measure.fit_transform([full_time_series])[0]
    
    return connectivity_matrix

def save_connectivity_results(conn_matrix, subject, labels_df, output_dir):
    """Save connectivity matrix and related information"""
    # Create subject-specific directory
    subj_dir = f"{output_dir}/{subject}"
    os.makedirs(subj_dir, exist_ok=True)
    
    # Save the raw matrix
    matrix_path = f"{subj_dir}/{subject}_object_connectivity.npy"
    np.save(matrix_path, conn_matrix)
    
    # Save as CSV with labels
    df_conn = pd.DataFrame(
        conn_matrix,
        index=labels_df['name'],
        columns=labels_df['name']
    )
    csv_path = f"{subj_dir}/{subject}_object_connectivity.csv"
    df_conn.to_csv(csv_path)
    
    # Create a specialized ROI-only matrix
    roi_mask = labels_df['type'] == 'specialized_roi'
    if roi_mask.any():
        roi_indices = np.where(roi_mask)[0]
        
        roi_matrix = conn_matrix[np.ix_(roi_indices, roi_indices)]
        roi_names = labels_df.loc[roi_indices, 'name'].tolist()
        
        df_roi_conn = pd.DataFrame(
            roi_matrix,
            index=roi_names,
            columns=roi_names
        )
        roi_csv_path = f"{subj_dir}/{subject}_object_roi_connectivity.csv"
        df_roi_conn.to_csv(roi_csv_path)
    
    logging.info(f"Saved connectivity results for {subject}")
    return matrix_path

def create_group_average(subjects, labels_df, output_dir):
    """Create group average connectivity matrix"""
    all_matrices = []
    
    for subject in subjects:
        matrix_path = f"{output_dir}/{subject}/{subject}_object_connectivity.npy"
        if os.path.exists(matrix_path):
            try:
                matrix = np.load(matrix_path)
                all_matrices.append(matrix)
            except Exception as e:
                logging.warning(f"Could not load matrix for {subject}: {str(e)}")
    
    if all_matrices:
        # Check if all matrices have the same shape
        shapes = [m.shape for m in all_matrices]
        if len(set(shapes)) > 1:
            logging.warning(f"Matrices have different shapes: {set(shapes)}")
            # Filter to keep only matrices with the most common shape
            most_common_shape = max(set(shapes), key=shapes.count)
            all_matrices = [m for m in all_matrices if m.shape == most_common_shape]
            logging.info(f"Keeping {len(all_matrices)} matrices with shape {most_common_shape}")
        
        # Compute average
        avg_matrix = np.mean(all_matrices, axis=0)
        
        # Save as numpy array
        avg_path = f"{output_dir}/group_average_object_connectivity.npy"
        np.save(avg_path, avg_matrix)
        
        # Save as CSV with labels
        if avg_matrix.shape[0] == len(labels_df):
            df_avg = pd.DataFrame(
                avg_matrix,
                index=labels_df['name'],
                columns=labels_df['name']
            )
            avg_csv_path = f"{output_dir}/group_average_object_connectivity.csv"
            df_avg.to_csv(avg_csv_path)
        else:
            logging.warning(f"Matrix size ({avg_matrix.shape[0]}) doesn't match labels ({len(labels_df)})")
            # Save without labels
            pd.DataFrame(avg_matrix).to_csv(f"{output_dir}/group_average_object_connectivity_no_labels.csv")
        
        logging.info(f"Created group average from {len(all_matrices)} subjects")
        return avg_path
    else:
        logging.warning("No matrices found for group average")
        return None

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup environment
    subs, raw_dir = setup_environment(args)
    
    # Load integrated parcellation
    parcellation_img, labels_df = load_integrated_parcellation(args.parcellation, args.labels)
    if parcellation_img is None or labels_df is None:
        logging.error("Failed to load parcellation. Exiting.")
        sys.exit(1)
    
    # Process subjects
    completed_subjects = []
    
    for ss in subs:
        try:
            # Create connectivity matrix for Object condition
            connectivity_matrix = create_connectivity_matrix(ss, parcellation_img, raw_dir)
            
            if connectivity_matrix is not None:
                # Save results
                save_path = save_connectivity_results(
                    connectivity_matrix, ss, labels_df, args.output_dir
                )
                
                # Verify the saved matrix
                try:
                    loaded_matrix = np.load(save_path)
                    if not np.allclose(connectivity_matrix, loaded_matrix):
                        logging.warning(f'Matrix verification failed for {ss}')
                    else:
                        completed_subjects.append(ss)
                        logging.info(f"Successfully processed subject {ss}")
                except Exception as e:
                    logging.warning(f"Could not verify matrix for {ss}: {str(e)}")
            
        except Exception as e:
            logging.error(f'Error processing subject {ss}: {str(e)}')
            continue
    
    # Create group average matrix
    if completed_subjects:
        create_group_average(completed_subjects, labels_df, args.output_dir)
    
    logging.info(f"Processing complete. Successfully processed {len(completed_subjects)} out of {len(subs)} subjects.")

if __name__ == "__main__":
    main()