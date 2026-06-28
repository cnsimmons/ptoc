"""
Compare sub-083 aCompCor PPI maps to the original (non-aCompCor) PPI maps.

Both are native anat space, so voxelwise correlation is valid.
High r (e.g. > .9) = aCompCor barely changes the result -> the argument
against re-running the whole pipeline.

Run:  python compare_ppi_acompcor.py
"""

import numpy as np
from nilearn import image

ss = 'sub-083'
deriv = f'/lab_data/behrmannlab/vlad/ptoc/{ss}/ses-01/derivatives'

pairs = [(roi, hemi) for roi in ['LO', 'pIPS'] for hemi in ['left', 'right']]

print(f'{"ROI":<8}{"hemi":<7}{"r":>8}{"n_vox":>10}')
for roi, hemi in pairs:
    orig_p   = f'{deriv}/fc/{ss}_{roi}_{hemi}_loc_ppi.nii.gz'
    acomp_p  = f'{deriv}/ppi/{ss}_{roi}_{hemi}_loc_ppi_acompcor.nii.gz'

    try:
        orig = image.load_img(orig_p)
        acomp = image.load_img(acomp_p)
    except Exception as e:
        print(f'{roi:<8}{hemi:<7}  missing file: {e}')
        continue

    # put aCompCor on the original grid if they differ
    if acomp.shape != orig.shape or not np.allclose(acomp.affine, orig.affine):
        acomp = image.resample_to_img(acomp, orig, interpolation='linear')

    o = orig.get_fdata().ravel()
    a = acomp.get_fdata().ravel()

    valid = np.isfinite(o) & np.isfinite(a) & ((o != 0) | (a != 0))
    if valid.sum() < 10:
        print(f'{roi:<8}{hemi:<7}  too few overlapping voxels')
        continue

    r = np.corrcoef(o[valid], a[valid])[0, 1]
    print(f'{roi:<8}{hemi:<7}{r:>8.3f}{valid.sum():>10}')