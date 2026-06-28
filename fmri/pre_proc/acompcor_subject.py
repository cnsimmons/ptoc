"""
aCompCor (textbook, Behzadi et al. 2007) — ONE subject, all runs.

Pipeline, per run:
  1. transform CSF (pve_0) and WM (pve_2) masks T1 -> native func space
     using FEAT's highres2example_func.mat (ref = example_func)
  2. threshold each at 0.99 and erode 1 voxel (in native func space)
  3. extract voxel timeseries from the RAW BOLD (FEAT's input) per mask
  4. PCA -> top 5 CSF + 5 WM components
  5. concatenate with the existing spike confound file (column-wise)
  6. write combined confound file (spikes + 10 aCompCor) per run

STOPS after writing the combined confound file. Does NOT run FEAT.

Run:  python acompcor_subject.py sub-083
"""

import os
import sys
import numpy as np
import nibabel as nib
from nilearn import image
from nilearn.maskers import NiftiMasker
from scipy import ndimage
from sklearn.decomposition import PCA

# ---- fixed parameters ----
raw_dir      = '/lab_data/behrmannlab/vlad/hemispace'
runs         = [1, 2, 3]
n_components = 5      # per tissue (5 CSF + 5 WM)
prob_thresh  = 0.99   # conservative tissue probability
wm_erode_iters = 2    # Behzadi 2007: WM eroded 2 voxels; CSF not eroded

def transform_mask(pve_path, ref, xfm, out_path):
    """flirt apply: T1 pve map -> native func grid (still continuous)."""
    import subprocess
    cmd = ['flirt', '-in', pve_path, '-ref', ref,
           '-applyxfm', '-init', xfm, '-out', out_path]
    print('  ' + ' '.join(cmd))
    subprocess.run(cmd, check=True)
    return out_path

def binarize_erode(prob_img_path, thresh, iters):
    """Threshold a (now native-space) prob map and erode; return mask img."""
    img = image.load_img(prob_img_path)
    data = img.get_fdata()
    binmask = (data >= thresh).astype(np.int16)
    if iters > 0:
        binmask = ndimage.binary_erosion(binmask, iterations=iters).astype(np.int16)
    return nib.Nifti1Image(binmask, img.affine, img.header), int(binmask.sum())

def extract_components(func_img, mask_img, n_comp):
    """Standardize+detrend voxel TS within mask, PCA, return (comps, var)."""
    masker = NiftiMasker(mask_img=mask_img, standardize=True, detrend=True)
    ts = masker.fit_transform(func_img)          # (nTR, nvox)
    if ts.shape[1] < n_comp:
        raise ValueError(f'mask has {ts.shape[1]} voxels; need >= {n_comp}')
    pca = PCA(n_components=n_comp)
    comps = pca.fit_transform(ts)                # (nTR, n_comp)
    return comps, pca.explained_variance_ratio_

def main(ss):
    anat_dir  = f'{raw_dir}/{ss}/ses-01/anat'
    func_base = f'{raw_dir}/{ss}/ses-01/derivatives/fsl/loc'
    func_raw  = f'{raw_dir}/{ss}/ses-01/func'
    out_dir   = f'{raw_dir}/{ss}/ses-01/derivatives/acompcor'
    os.makedirs(out_dir, exist_ok=True)

    csf_pve = f'{anat_dir}/{ss}_fast_pve_0.nii.gz'
    wm_pve  = f'{anat_dir}/{ss}_fast_pve_2.nii.gz'
    for p in (csf_pve, wm_pve):
        if not os.path.exists(p):
            print(f'ERROR: missing FAST output {p} (run FAST first)')
            sys.exit(1)

    for rn in runs:
        print(f'\n=== run {rn} ===')
        feat = f'{func_base}/run-0{rn}/1stLevel.feat'
        reg  = f'{feat}/reg'
        xfm  = f'{reg}/highres2example_func.mat'
        ref  = f'{reg}/example_func.nii.gz'
        bold = f'{func_raw}/{ss}_ses-01_task-loc_run-0{rn}_bold.nii.gz'
        spikes = f'{func_base}/run-0{rn}/{ss}_ses-01_task-loc_run-0{rn}_bold_spikes.txt'

        for need in (xfm, ref, bold):
            if not os.path.exists(need):
                print(f'  missing {need}; skipping run {rn}')
                break
        else:
            # 1. transform masks T1 -> native func
            csf_func = transform_mask(csf_pve, ref, xfm,
                          f'{out_dir}/{ss}_run-0{rn}_csf_func.nii.gz')
            wm_func  = transform_mask(wm_pve, ref, xfm,
                          f'{out_dir}/{ss}_run-0{rn}_wm_func.nii.gz')

            # 2. threshold + erode (native space)
            # Behzadi 2007: WM eroded 2 voxels; CSF NOT eroded (regions too small)
            csf_mask, n_csf = binarize_erode(csf_func, prob_thresh, iters=0)
            wm_mask,  n_wm  = binarize_erode(wm_func,  prob_thresh, iters=wm_erode_iters)
            print(f'  CSF voxels={n_csf}, WM voxels={n_wm}')

            # 3-4. extract from RAW BOLD + PCA
            bold_img = image.load_img(bold)
            print(f'  BOLD {bold_img.shape}')
            csf_c, csf_v = extract_components(bold_img, csf_mask, n_components)
            wm_c,  wm_v  = extract_components(bold_img, wm_mask,  n_components)
            print(f'  CSF var (5)={csf_v.sum():.2%}, WM var (5)={wm_v.sum():.2%}')

            acompcor = np.hstack([csf_c, wm_c])          # (nTR, 10)

            # 5. concat with existing spikes (if present)
            if os.path.exists(spikes):
                spk = np.loadtxt(spikes)
                if spk.ndim == 1:
                    spk = spk.reshape(-1, 1)
                if spk.shape[0] != acompcor.shape[0]:
                    print(f'  WARNING: spikes rows={spk.shape[0]} '
                          f'!= aCompCor rows={acompcor.shape[0]}')
                combined = np.hstack([spk, acompcor])
                print(f'  spikes cols={spk.shape[1]} + aCompCor 10 '
                      f'= {combined.shape[1]} cols')
            else:
                combined = acompcor
                print('  no spikes file found; writing aCompCor only (10 cols)')

            # 6. write combined confound file (tab-sep, no header — FEAT format)
            out = f'{out_dir}/{ss}_run-0{rn}_confounds_combined.txt'
            np.savetxt(out, combined, fmt='%.6f', delimiter='\t')
            print(f'  wrote {combined.shape} -> {out}')

    print('\nDone. Combined confound files written. FEAT NOT run.')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python acompcor_subject.py <subject>  (e.g. sub-083)')
        sys.exit(1)
    main(sys.argv[1])