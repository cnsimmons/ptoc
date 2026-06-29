"""
Verify the sub-083 PPI comparison is honest: confirm the aCompCor map and the
original map are DIFFERENT files with small-but-nonzero differences.

Rules out the failure mode where r~1.0 just means a file was compared to itself.

For each ROI/hemi prints:
  - whether both files exist, their sizes, and md5 (different md5 = different files)
  - difference-map stats (min/max/mean-abs): want small but NOT all-zero
  - in-brain correlation (masked to the subject brain), as a stricter r than
    the nonzero-union version

Run:  python verify_ppi_compare.py
"""

import os
import hashlib
import numpy as np
import nibabel as nib
from nilearn import image

ss = 'sub-083'
deriv = f'/lab_data/behrmannlab/vlad/ptoc/{ss}/ses-01/derivatives'
brain_mask_p = f'/lab_data/behrmannlab/vlad/hemispace/{ss}/ses-01/anat/{ss}_ses-01_T1w_brain_mask.nii.gz'

pairs = [(roi, hemi) for roi in ['LO', 'pIPS'] for hemi in ['left', 'right']]


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:12]


# optional brain mask for a stricter, in-brain correlation
mask = None
if os.path.exists(brain_mask_p):
    mask = image.load_img(brain_mask_p)
else:
    print(f'(no brain mask at {brain_mask_p}; falling back to nonzero-union)\n')

for roi, hemi in pairs:
    orig_p  = f'{deriv}/fc/{ss}_{roi}_{hemi}_loc_ppi.nii.gz'
    acomp_p = f'{deriv}/ppi/{ss}_{roi}_{hemi}_loc_ppi_acompcor.nii.gz'

    print(f'==== {roi} {hemi} ====')
    if not os.path.exists(orig_p):
        print(f'  MISSING original: {orig_p}\n'); continue
    if not os.path.exists(acomp_p):
        print(f'  MISSING acompcor: {acomp_p}\n'); continue

    # 1. different files?
    so, sa = os.path.getsize(orig_p), os.path.getsize(acomp_p)
    ho, ha = md5(orig_p), md5(acomp_p)
    print(f'  original : {so:>10} bytes  md5 {ho}')
    print(f'  acompcor : {sa:>10} bytes  md5 {ha}')
    if orig_p == acomp_p or ho == ha:
        print('  !! SAME FILE / identical bytes -> comparison is meaningless\n')
        continue
    print('  files differ: OK')

    # 2. load, align acompcor to original grid if needed
    orig = image.load_img(orig_p)
    acomp = image.load_img(acomp_p)
    if acomp.shape != orig.shape or not np.allclose(acomp.affine, orig.affine):
        acomp = image.resample_to_img(acomp, orig, interpolation='linear')

    o = orig.get_fdata()
    a = acomp.get_fdata()

    # 3. difference-map stats: want small but nonzero
    diff = a - o
    print(f'  diff  min/max = {diff.min():.4g} / {diff.max():.4g}')
    print(f'  diff  mean|.| = {np.abs(diff).mean():.4g}')
    if np.all(diff == 0):
        print('  !! difference is EXACTLY zero everywhere -> same data\n')
        continue
    print('  difference small but nonzero: OK')

    # 4. stricter in-brain correlation
    if mask is not None:
        m = (image.resample_to_img(mask, orig, interpolation='nearest')
             .get_fdata() > 0)
        region = 'in-brain'
    else:
        m = (o != 0) | (a != 0)
        region = 'nonzero-union'
    ov, av = o[m], a[m]
    valid = np.isfinite(ov) & np.isfinite(av)
    r = np.corrcoef(ov[valid], av[valid])[0, 1]
    print(f'  r ({region}, n={valid.sum()}) = {r:.4f}\n')