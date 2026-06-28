# `fmri/pre_proc/` — Preprocessing & Registration

Preprocessing and registration scripts for the ptoc project (Experiment 1,
object/scramble localizer `loc`). Functional analyses live elsewhere; this
folder takes raw data through motion handling, the FEAT GLM, and registration
into the spaces the connectivity analyses expect.

Two data trees are used throughout:
- `raw_dir` = `/lab_data/behrmannlab/vlad/hemispace` — anat, func, FEAT dirs
- `data_dir` = `/lab_data/behrmannlab/vlad/ptoc` — non-anat derivatives

---

## Pipeline order

1. **`copy_loc.py`**
   Copies raw anat + localizer func from the source archive into
   `hemispace`, renames sessions to `ses-01`, and builds per-condition
   covariate `.txt` files (`catloc_{sub}_run-0{run}_{cond}.txt`) from each
   run's `events.tsv`. Conditions: Face, House, Object, Scramble, Word.

2. **`preprocess.py {sub}`**
   Brain extraction (`bet -R -B`) on the T1, then per run computes motion
   outliers with `fsl_motion_outliers` → writes the spike confound file
   `..._bold_spikes.txt`. This spike file is what FEAT later reads as its
   confound EV.

2b. **`run_fast_controls.sh`** *(NEW — for aCompCor)*
   Runs FSL `FAST` (`-t 1 -n 3`) on each control's brain-extracted T1,
   looping over `group=='control'` in `sub_info.csv` and skipping subjects
   already segmented. Outputs tissue probability maps in each anat folder:
   `{sub}_fast_pve_0` (CSF), `_pve_1` (GM), `_pve_2` (WM). These feed the
   aCompCor step below. Run once per subject.

3. **FEAT 1st-level GLM** *(launched via `run_job.py`, not in this folder)*
   `feat .../run-0{run}/1stLevel{suf}.fsf`. The design references the spike
   file as `confoundev_files(1)`. Produces `filtered_func_data.nii.gz`,
   `stats/`, and `reg/` (registration matrices + `example_func`).

4. **`register_1stlevel.py {sub}`**
   Registers `filtered_func_data` → `filtered_func_data_reg.nii.gz` and the
   zstat of interest, using `reg/example_func2standard.mat`, with the
   subject's **T1 brain** as reference. Output therefore sits in subject-anat
   space (≈176×256×256, 1mm) — this is the space the connectivity analyses
   read from.

5. **`register_highlevel.py`**
   Registers HighLevel `.gfeat` cope zstats to MNI using `anat2stand.mat`.

6. **`1stLevel_zstat2anat_mni.py`**
   Two-step registration of 1st-level zstats: func→anat
   (`example_func2highres.mat`) then anat→MNI (`anat2stand.mat`).

7. **`register_mirror.py`**
   Anatomical mirroring / hemisphere masks for patients (mirror brain, hemi
   masks), MNI registration, and parcel registration into subject space
   (`mni2anat.mat`). Used for ROI parcel placement.

8. **`transform_func_standard.sh`**
   Transforms `filtered_func_data_reg` → MNI 2mm standard
   (`filtered_func_run-0{run}_standard.nii.gz`) using
   `ptoc/.../derivatives/anat2mni.mat`. Used for the Schaefer-atlas
   connectivity-matrix analysis, which needs 2mm standard space.

**Helper:** `change_permissions.py` — file-permission housekeeping, not part
of the analysis flow.

---

## aCompCor (NEW — proposed addition, not yet integrated)

Added in response to Reviewer 2.1, which requested aCompCor denoising. This
is **new to the pipeline** — no prior version of this project, and no related
lab pipeline, used aCompCor. It is documented here so the integration point
is explicit.

Intended approach (textbook Behzadi et al., 2007): 5 CSF + 5 WM principal
components, regressed in the FEAT GLM as additional confound EVs alongside the
existing spike columns.

Required steps (per subject):

1. **`run_fast_controls.sh`** — FAST segmentation (step 2b above). *(Done for all controls.)*

2. **`acompcor_subject.py {sub}`** — per run: transform CSF (`pve_0`) / WM
   (`pve_2`) maps T1 → native func space using the existing
   `1stLevel.feat/reg/highres2example_func.mat` (ref = `reg/example_func`);
   threshold 0.99; erode WM 2 voxels, CSF not eroded (Behzadi 2007); extract
   from raw BOLD; PCA → 5 CSF + 5 WM; concatenate with the existing spike
   confound file → `acompcor/{sub}_run-0{rn}_confounds_combined.txt`. Stops
   before FEAT. *(pve_0=CSF, pve_2=WM confirmed in FSLeyes — standard FAST
   ordering.)*

3. **`make_acompcor_fsf.py {sub}`** — per run: copy `1stLevel.fsf` →
   `1stLevel_acompcor.fsf`, changing only `outputdir` (→ `1stLevel_acompcor`)
   and `confoundev_files(1)` (→ combined confound file). Originals untouched;
   results go to a parallel `_acompcor` path so nothing is overwritten.

4. **Re-run FEAT GLM** with the `_acompcor` design (via `run_job.py`,
   `suf='_acompcor'`) → new `1stLevel_acompcor.feat`.

5. **`register_1stlevel.py`** on the new FEAT output → aCompCor-cleaned
   `filtered_func_data_reg` for analysis.

### Note
- Current plan is one subject (`sub-083`), replicating Fig 3C/D with aCompCor
  to assess whether maps change, as the basis for arguing against full
  reprocessing. Whether this satisfies R2.1, or full-sample reprocessing is
  required, is a group decision.