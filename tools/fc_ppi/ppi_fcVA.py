def conduct_analyses():
    """Conduct PPI analyses for all subjects and ROIs"""
    logger = setup_logging()
    
    for ss in subs:
        logger.info(f"Processing subject: {ss}")
        
        temp_dir = f'{raw_dir}/{ss}/ses-01/derivatives/fsl/toolloc'
        mask_path = f'{raw_dir}/{ss}/ses-01/anat/{ss}_ses-01_T1w_brain_mask.nii.gz'
        out_dir = f'/user_data/csimmon2/temp_derivatives/{ss}/ses-01/derivatives'
        os.makedirs(f'{out_dir}/fc', exist_ok=True)
        
        roi_coords = pd.read_csv(f'{output_dir}/roi_coordinates.csv')
        
        try:
            whole_brain_mask = nib.load(mask_path)
            brain_masker = NiftiMasker(whole_brain_mask, standardize=True)
            
            for roi in rois:
                for hemi in hemispheres:
                    hemi_prefix = hemi[0]
                    logger.info(f"Processing {roi} {hemi}")
                    
                    fc_file = f'{out_dir}/fc/{ss}_{roi}_{hemi}_ToolLoc_fc.nii.gz'
                    if os.path.exists(fc_file):
                        logger.info(f"Skipping {ss} {roi} {hemi} - already processed")
                        continue
                    
                    all_runs = []
                    
                    for rcn, rc in enumerate(run_combos):
                        roi_run = rc[0]
                        analysis_run = rc[1]
                        
                        try:
                            curr_coords = roi_coords[
                                (roi_coords['subject'] == ss) &
                                (roi_coords['run_combo'] == rcn) & 
                                (roi_coords['roi'] == f"{hemi_prefix}{roi}") &
                                (roi_coords['hemisphere'] == hemi_prefix)
                            ]
                            
                            if curr_coords.empty:
                                continue
                                
                            coords = [
                                curr_coords['x'].values[0],
                                curr_coords['y'].values[0],
                                curr_coords['z'].values[0]
                            ]
                            
                            img = image.clean_img(
                                image.load_img(f'{temp_dir}/run-0{analysis_run}/1stLevel.feat/filtered_func_data_reg.nii.gz'),
                                standardize=True
                            )
                            
                            # Get physiological regressor
                            phys = extract_roi_sphere(img, coords)

                            # Get psychological regressor 
                            psy = make_psy_cov(analysis_run, ss)

                            # Create PPI regressor
                            ppi_regressor = phys * psy
                            
                            # Include both main effects and PPI as confounds
                            confounds = pd.DataFrame({
                                'psy': psy[:,0],
                                'phys': phys[:,0],
                                'ppi': ppi_regressor[:,0]
                            })

                            # Get brain timeseries controlling for all effects
                            brain_time_series = brain_masker.fit_transform(img, confounds=confounds)
                            
                            # Get correlations 
                            correlations = np.dot(brain_time_series.T, ppi_regressor) / ppi_regressor.shape[0]
                            correlations = np.arctanh(correlations.ravel())
                            correlation_img = brain_masker.inverse_transform(correlations)
                            
                            all_runs.append(correlation_img)

                        except Exception as e:
                            logger.error(f"Error in run combo {rc}: {str(e)}")
                            continue
                    
                    if all_runs:
                        mean_fc = image.mean_img(all_runs)
                        nib.save(mean_fc, fc_file)
        
        except Exception as e:
            logger.error(f"Error processing subject {ss}: {str(e)}")
            continue