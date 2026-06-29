"""
Exp1 PPI on aCompCor-preprocessed data — sub-083 comparison.

Same as exp1_fc_ppi.py PPI, with ONE change: timeseries are read from
1stLevel_acompcor.feat instead of 1stLevel.feat. The ROI is unchanged —
it uses the existing sphere_coords_hemisphere.csv (original definition),
so this isolates exactly what aCompCor changes in the connectivity.

Outputs are suffixed _acompcor so they sit beside the originals for
side-by-side comparison; nothing original is overwritten.

Run:  python exp1_ppi_acompcor.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image, input_data
from nilearn.maskers import NiftiMasker
from nilearn.glm.first_level import compute_regressor

curr_dir = '/user_data/csimmon2/git_repos/ptoc'
sys.path.insert(0, curr_dir)
import ptoc_params as params

study = 'ptoc'
study_dir = f'/lab_data/behrmannlab/vlad/{study}'
raw_dir = params.raw_dir

subs = ['sub-083']                 # comparison subject only
rois = ['LO', 'pIPS', 'PFS', 'V1']
hemispheres = ['left', 'right']

feat_suf = '_acompcor'             # <-- the only pipeline change
out_suf = '_acompcor'             # keeps outputs separate from originals

run_num = 3
run_combos = [[rn1, rn2] for rn1 in range(1, run_num + 1)
              for rn2 in range(rn1 + 1, run_num + 1)]


def extract_roi_sphere(img, coords):
    roi_masker = input_data.NiftiSpheresMasker([tuple(coords)], radius=6)
    seed_time_series = roi_masker.fit_transform(img)
    return np.mean(seed_time_series, axis=1).reshape(-1, 1)


def make_psy_cov(runs, ss):
    cov_dir = f'{raw_dir}/{ss}/ses-01/covs'
    vols, tr = 184, 2.0
    times = np.arange(0, vols * tr, tr)
    full_cov = pd.DataFrame(columns=['onset', 'duration', 'value'])

    for rn in runs:
        ss_num = ss.split('-')[1]
        obj = f'{cov_dir}/catloc_{ss_num}_run-0{rn}_Object.txt'
        scr = f'{cov_dir}/catloc_{ss_num}_run-0{rn}_Scramble.txt'
        if not os.path.exists(obj) or not os.path.exists(scr):
            print(f'  covariate file not found for run {rn}')
            continue
        obj_cov = pd.read_csv(obj, sep='\t', header=None,
                              names=['onset', 'duration', 'value'])
        scr_cov = pd.read_csv(scr, sep='\t', header=None,
                              names=['onset', 'duration', 'value'])
        scr_cov['value'] *= -1
        full_cov = pd.concat([full_cov, obj_cov, scr_cov])

    full_cov = full_cov.sort_values(by=['onset']).reset_index(drop=True)
    cov = full_cov.to_numpy()
    cov = cov[cov[:, 0] < times[-1]]
    if cov.shape[0] == 0:
        print('  no valid covariate data; returning zeros')
        return np.zeros((vols, 1))
    psy, _ = compute_regressor(cov.T, 'spm', times)
    return psy


def conduct_analyses():
    for ss in subs:
        print(f'Processing subject: {ss}')
        sub_dir = f'{study_dir}/{ss}/ses-01/'
        roi_dir = f'{sub_dir}derivatives/rois'
        temp_dir = f'{raw_dir}/{ss}/ses-01/derivatives/fsl/loc'

        # existing ROI coordinates (original definition, unchanged)
        roi_coords = pd.read_csv(f'{roi_dir}/spheres/sphere_coords_hemisphere.csv')

        out_dir = f'{study_dir}/{ss}/ses-01/derivatives'
        os.makedirs(f'{out_dir}/ppi', exist_ok=True)

        mask_path = f'{raw_dir}/{ss}/ses-01/anat/{ss}_ses-01_T1w_brain_mask.nii.gz'
        brain_masker = NiftiMasker(nib.load(mask_path),
                                   smoothing_fwhm=0, standardize=True)

        tsk = 'loc'
        for rr in rois:
            for hemi in hemispheres:
                t0 = time.time()
                print(f'  {rr} {hemi}')
                ppi_file = f'{out_dir}/ppi/{ss}_{rr}_{hemi}_{tsk}_ppi{out_suf}.nii.gz'
                if os.path.exists(ppi_file):
                    print(f'    exists, skipping: {ppi_file}')
                    continue

                all_runs_ppi = []
                for rcn, rc in enumerate(run_combos):
                    cc = roi_coords[(roi_coords['index'] == rcn) &
                                    (roi_coords['task'] == tsk) &
                                    (roi_coords['roi'] == rr) &
                                    (roi_coords['hemisphere'] == hemi)]
                    if cc.empty:
                        print(f'    no coords for run combo {rc}')
                        continue
                    coords = cc[['x', 'y', 'z']].values.tolist()[0]

                    # read from the aCompCor FEAT dir
                    paths = [f'{temp_dir}/run-0{rn}/1stLevel{feat_suf}.feat/'
                             f'filtered_func_data_reg.nii.gz' for rn in rc]
                    missing = [p for p in paths if not os.path.exists(p)]
                    if missing:
                        print(f'    missing aCompCor func: {missing}')
                        continue
                    filtered = [image.clean_img(image.load_img(p), standardize=True)
                                for p in paths]
                    img4d = image.concat_imgs(filtered)

                    phys = extract_roi_sphere(img4d, coords)
                    if phys.shape[0] > 184 * len(rc):
                        phys = phys[:184 * len(rc)]

                    psy = make_psy_cov(rc, ss)
                    if psy.shape[0] > phys.shape[0]:
                        psy = psy[:phys.shape[0]]
                    elif psy.shape[0] < phys.shape[0]:
                        phys = phys[:psy.shape[0]]

                    brain_ts = brain_masker.fit_transform(img4d)
                    brain_ts = brain_ts[:phys.shape[0]]

                    ppi_reg = phys * psy
                    corr = np.dot(brain_ts.T, ppi_reg) / ppi_reg.shape[0]
                    corr = np.arctanh(corr.ravel())
                    all_runs_ppi.append(brain_masker.inverse_transform(corr))

                if all_runs_ppi:
                    nib.save(image.mean_img(all_runs_ppi), ppi_file)
                    print(f'    saved {ppi_file} ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    conduct_analyses()