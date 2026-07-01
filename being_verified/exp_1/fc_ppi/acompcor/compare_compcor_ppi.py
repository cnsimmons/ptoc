"""
Compare aCompCor vs original PPI maps for 5 completed subjects.

Native-space voxelwise correlation — no MNI or thresholding needed.
High r (>0.99) = aCompCor barely changes the result.

Outputs:
  - per-subject/ROI/hemi correlation table (printed + CSV)
  - summary stats for the reviewer response

Run:  python compare_compcor_5subs.py
"""

import os
import hashlib
import numpy as np
import nibabel as nib
from nilearn import image
import pandas as pd

# --- CONFIG ---
subs = ['sub-025', 'sub-038', 'sub-083', 'sub-093', 'sub-107']
rois = ['LO', 'pIPS']
hemis = ['left', 'right']

study_dir = '/lab_data/behrmannlab/vlad/ptoc'
raw_dir = '/lab_data/behrmannlab/vlad/hemispace'

results = []

def md5_short(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:12]


for ss in subs:
    deriv = f'{study_dir}/{ss}/ses-01/derivatives'
    mask_path = f'{raw_dir}/{ss}/ses-01/anat/{ss}_ses-01_T1w_brain_mask.nii.gz'

    # load brain mask once per subject
    mask = None
    if os.path.exists(mask_path):
        mask = image.load_img(mask_path)

    for roi in rois:
        for hemi in hemis:
            orig_p  = f'{deriv}/fc/{ss}_{roi}_{hemi}_loc_ppi.nii.gz'
            acomp_p = f'{deriv}/ppi/{ss}_{roi}_{hemi}_loc_ppi_acompcor.nii.gz'

            row = {'subject': ss, 'roi': roi, 'hemi': hemi}

            # check existence
            if not os.path.exists(orig_p):
                row['status'] = 'missing_original'
                results.append(row)
                print(f'{ss} {roi} {hemi}: MISSING original')
                continue
            if not os.path.exists(acomp_p):
                row['status'] = 'missing_acompcor'
                results.append(row)
                print(f'{ss} {roi} {hemi}: MISSING acompcor')
                continue

            # verify different files
            h_orig, h_acomp = md5_short(orig_p), md5_short(acomp_p)
            if h_orig == h_acomp:
                row['status'] = 'identical_files'
                results.append(row)
                print(f'{ss} {roi} {hemi}: IDENTICAL FILES — comparison meaningless')
                continue

            # load images
            orig = image.load_img(orig_p)
            acomp = image.load_img(acomp_p)

            # resample if grids differ
            if acomp.shape != orig.shape or not np.allclose(acomp.affine, orig.affine):
                acomp = image.resample_to_img(acomp, orig, interpolation='linear')

            o = orig.get_fdata()
            a = acomp.get_fdata()

            # in-brain mask
            if mask is not None:
                m = (image.resample_to_img(mask, orig, interpolation='nearest')
                     .get_fdata() > 0)
            else:
                m = (o != 0) | (a != 0)

            ov, av = o[m], a[m]
            valid = np.isfinite(ov) & np.isfinite(av)
            n_vox = valid.sum()

            if n_vox < 10:
                row['status'] = 'too_few_voxels'
                results.append(row)
                print(f'{ss} {roi} {hemi}: too few overlapping voxels')
                continue

            r = np.corrcoef(ov[valid], av[valid])[0, 1]
            diff = av[valid] - ov[valid]

            row.update({
                'status': 'ok',
                'r': r,
                'n_voxels': int(n_vox),
                'diff_mean_abs': np.abs(diff).mean(),
                'diff_max_abs': np.abs(diff).max(),
            })
            results.append(row)
            print(f'{ss} {roi} {hemi}: r = {r:.4f}  (n={n_vox}, mean|diff|={np.abs(diff).mean():.4g})')

# --- SUMMARY ---
df = pd.DataFrame(results)
ok = df[df['status'] == 'ok']

print(f'\n{"="*60}')
print(f'SUMMARY: {len(ok)} of {len(df)} comparisons completed')
print(f'{"="*60}')

if len(ok) > 0:
    print(f'  r  mean = {ok["r"].mean():.4f}')
    print(f'  r  min  = {ok["r"].min():.4f}')
    print(f'  r  max  = {ok["r"].max():.4f}')
    print(f'  r  std  = {ok["r"].std():.4f}')
    print(f'  mean |diff| across subjects = {ok["diff_mean_abs"].mean():.4g}')
    print(f'\nPer-ROI summary:')
    for roi in rois:
        roi_ok = ok[ok['roi'] == roi]
        if len(roi_ok) > 0:
            print(f'  {roi}: mean r = {roi_ok["r"].mean():.4f} (n={len(roi_ok)})')

missing = df[df['status'] != 'ok']
if len(missing) > 0:
    print(f'\nIssues ({len(missing)}):')
    for _, row in missing.iterrows():
        print(f'  {row["subject"]} {row["roi"]} {row["hemi"]}: {row["status"]}')

# save CSV
out_csv = f'{results_dir}/acompcor_ppi_comparison_5subs.csv' if 'results_dir' in dir() else '/user_data/csimmon2/git_repos/ptoc/results/acompcor_ppi_comparison_5subs.csv'
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
df.to_csv(out_csv, index=False)
print(f'\nSaved: {out_csv}')