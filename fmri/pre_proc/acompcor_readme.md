# aCompCor — README

## Why this exists

Reviewer 2 (comment R2.1) requested that the data be preprocessed with
aCompCor to remove potential physiological/noise contaminants (CSF and white
matter signal) before the connectivity analyses, in order to rule out the
possibility that the observed dorsal–ventral connectivity overlap is driven by
shared noise.

aCompCor is **new to this project** — no prior version of this pipeline, and no
related lab pipeline, used it. It is added here as an extra set of nuisance
regressors in the existing FEAT GLM. Nothing else in the pipeline changes: the
aCompCor components are simply appended to the confound file FEAT already
reads, and the rest of preprocessing and analysis runs exactly as before.

## Method

Textbook anatomical CompCor (Behzadi et al., 2007): principal components of the
BOLD signal within CSF and white-matter masks are used as nuisance regressors.
5 CSF + 5 WM components (10 total), following common practice (Muschelli et
al., 2014). These 10 components are added to the existing spike (motion-outlier)
confound regressors and entered into the FEAT first-level GLM as confound EVs.

## Exact steps taken (per subject)

All work was done on a parallel `_acompcor` path; original `.fsf`, `.feat`, and
registered outputs were never overwritten.

1. **Tissue segmentation — `run_fast_controls.sh`**
   FSL `FAST` (`-t 1 -n 3`) on each control's brain-extracted T1
   (`{sub}_ses-01_T1w_brain.nii.gz`), looping over `group=='control'` in
   `sub_info.csv`. Produces tissue probability maps in the anat folder:
   `{sub}_fast_pve_0` (CSF), `_pve_1` (GM), `_pve_2` (WM). Masks were inspected
   in FSLeyes against the T1; tissue identity confirmed (pve_0 = CSF in the
   ventricles/sulci, pve_2 = WM in the interior — standard FAST ordering).

2. **Build aCompCor regressors — `acompcor_subject.py {sub}`**
   Per run:
   - Transform CSF (`pve_0`) and WM (`pve_2`) maps from T1 into native
     functional space using the existing FEAT registration
     (`1stLevel.feat/reg/highres2example_func.mat`, ref =
     `reg/example_func`). No new registration was computed.
   - Threshold each map at a partial-volume fraction of 0.99.
   - Erode the WM mask by 2 voxels (minimize gray-matter partial volume); the
     CSF mask is **not** eroded (CSF regions are small; Behzadi 2007).
   - Extract voxel timeseries from the raw BOLD within each mask; run PCA;
     keep the top 5 components per tissue.
   - Concatenate the 10 aCompCor columns with the existing spike confound file
     → `acompcor/{sub}_run-0{rn}_confounds_combined.txt`.

   Transformed masks were inspected in FSLeyes (correct tissue, aligned to the
   BOLD grid). Voxel counts and component variance were checked per run
   (e.g. sub-083: CSF ≈ 520–560 voxels, WM ≈ 1850–1940).

3. **Create aCompCor design files — `make_acompcor_fsf.py {sub}`**
   Per run, copy `1stLevel.fsf` → `1stLevel_acompcor.fsf`, changing only two
   lines: `fmri(outputdir)` (→ `1stLevel_acompcor`) and `confoundev_files(1)`
   (→ the combined confound file). All other model settings identical
   (verified by `diff`).

4. **Re-run FEAT GLM**
   `feat .../run-0{rn}/1stLevel_acompcor.fsf` per run → `1stLevel_acompcor.feat`.
   The aCompCor + spike regressors are removed in the GLM here.

5. **Register the new output — `register_1stlevel_acompcor.py {sub}`**
   Registers `filtered_func_data` (and the zstat of interest) from each
   `1stLevel_acompcor.feat` into anat space using
   `reg/example_func2standard.mat`, producing `filtered_func_data_reg.nii.gz`.
   This is the aCompCor-cleaned data the connectivity analyses read.

After step 5, the aCompCor-cleaned data is ready and the downstream analyses
(FC, PPI, partial correlation, GCA) run unchanged on it. FDR correction is
applied at the connectivity/group-map stage, as in the original analyses — it
is not part of preprocessing.

## Status

Pilot/sanity check on one subject (`sub-083`), to replicate Fig 3C/D with
aCompCor and assess whether the connectivity maps change. Whether this
satisfies R2.1 or full-sample reprocessing is required is a group decision.

## References
- Behzadi Y, Restom K, Liau J, Liu TT (2007). A component based noise
  correction method (CompCor) for BOLD and perfusion based fMRI. NeuroImage
  37(1):90–101.
- Muschelli J et al. (2014). Reduction of motion-related artifacts in resting
  state fMRI using aCompCor. NeuroImage 96:22–35.