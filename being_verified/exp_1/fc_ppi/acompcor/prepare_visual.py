# Surface visualization: Original vs aCompCor PPI overlap
# Inflated fsaverage, z > 2.58, ~45-degree views
import os
import numpy as np
import nibabel as nib
from nilearn import plotting, datasets, image, surface
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

results_dir = '/user_data/csimmon2/git_repos/ptoc/results'
group_out_dir = f'{results_dir}/group_averages'
out_dir = f'{results_dir}/acompcor_comparison'
os.makedirs(out_dir, exist_ok=True)

display_thresh = 2.58

# Load fsaverage inflated
fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage')

# ~45 degree views: (elevation, azimuth)
views_left =  [(30, 210), (-30, 30)]   # lateral-dorsal, medial-ventral
views_right = [(30, 330), (-30, 150)]  # lateral-dorsal, medial-ventral
view_labels = ['lateral', 'medial']

# --- Helpers ---
def load_and_average(roi, suffix):
    imgs = []
    for hemi in ['left', 'right']:
        p = f'{group_out_dir}/{roi}_{hemi}_ppi{suffix}.nii.gz'
        if os.path.exists(p):
            imgs.append(image.load_img(p))
        else:
            print(f"Missing: {p}")
    if len(imgs) == 2:
        return image.mean_img(imgs)
    elif len(imgs) == 1:
        return imgs[0]
    return None

def apply_thresh(img, thresh):
    data = img.get_fdata().copy()
    data[data < thresh] = 0
    return nib.Nifti1Image(data, img.affine)

def make_overlap(pips_img, lo_img, thresh):
    p = apply_thresh(pips_img, thresh)
    l = apply_thresh(lo_img, thresh)
    pbin = (p.get_fdata() > 0).astype(int)
    lbin = (l.get_fdata() > 0).astype(int)
    combined = np.zeros_like(pbin, dtype=float)
    combined[(pbin == 1) & (lbin == 0)] = 1  # dorsal only
    combined[(pbin == 1) & (lbin == 1)] = 2  # overlap
    combined[(pbin == 0) & (lbin == 1)] = 3  # ventral only
    n_d = np.sum(combined == 1)
    n_o = np.sum(combined == 2)
    n_v = np.sum(combined == 3)
    print(f"  Dorsal-only: {n_d}, Overlap: {n_o}, Ventral-only: {n_v}")
    return nib.Nifti1Image(combined, p.affine)

def plot_overlap_row(combined_img, axes_row, cmap, label_prefix=""):
    """Plot one row (LH lateral, LH medial, RH lateral, RH medial)"""
    for col, (hemi_name, views) in enumerate([('left', views_left), ('right', views_right)]):
        infl = fsaverage[f'infl_{hemi_name}']
        sulc = fsaverage[f'sulc_{hemi_name}']
        pial = fsaverage[f'pial_{hemi_name}']
        texture = surface.vol_to_surf(combined_img, pial)
        
        for v_idx, view in enumerate(views):
            ax_idx = col * 2 + v_idx
            plotting.plot_surf_stat_map(
                infl, texture,
                hemi=hemi_name,
                view=view,
                colorbar=False,
                cmap=cmap,
                threshold=0.5,
                vmax=3.5,
                bg_map=sulc,
                axes=axes_row[ax_idx],
                symmetric_cbar=False
            )

# --- Load both pipelines ---
print("Loading original maps...")
pips_orig = load_and_average('pIPS', '_thresh')
lo_orig = load_and_average('LO', '_thresh')

print("Loading aCompCor maps...")
pips_acomp = load_and_average('pIPS', '_acompcor_thresh')
lo_acomp = load_and_average('LO', '_acompcor_thresh')

if any(x is None for x in [pips_orig, lo_orig, pips_acomp, lo_acomp]):
    print("ERROR: Missing thresholded maps for one or both pipelines.")
    exit(1)

print("\nOriginal overlap:")
overlap_orig = make_overlap(pips_orig, lo_orig, display_thresh)
print("aCompCor overlap:")
overlap_acomp = make_overlap(pips_acomp, lo_acomp, display_thresh)

# --- Side-by-side figure ---
cmap_overlap = ListedColormap(['#4ac0c0', '#9467bd', '#ff9b83'])

fig, axes = plt.subplots(2, 4, figsize=(20, 10),
                          subplot_kw={'projection': '3d'})

print("\nPlotting original (top row)...")
plot_overlap_row(overlap_orig, axes[0], cmap_overlap)
axes[0][0].set_title('LH lateral', fontsize=10)
axes[0][1].set_title('LH medial', fontsize=10)
axes[0][2].set_title('RH lateral', fontsize=10)
axes[0][3].set_title('RH medial', fontsize=10)

print("Plotting aCompCor (bottom row)...")
plot_overlap_row(overlap_acomp, axes[1], cmap_overlap)

# Row labels
fig.text(0.02, 0.72, 'Original', fontsize=14, fontweight='bold', rotation=90, va='center')
fig.text(0.02, 0.28, 'aCompCor', fontsize=14, fontweight='bold', rotation=90, va='center')

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#4ac0c0', label='Dorsal only'),
    mpatches.Patch(facecolor='#9467bd', label='Overlap'),
    mpatches.Patch(facecolor='#ff9b83', label='Ventral only')
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3,
           fontsize=12, frameon=True, bbox_to_anchor=(0.5, 0.01))

fig.suptitle('PPI Overlap: Original vs aCompCor (z > 2.58)', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0.04, 0.05, 1, 0.95])
plt.savefig(f'{out_dir}/surface_overlap_comparison.png', dpi=300, bbox_inches='tight')
plt.close('all')
print(f"\nSaved: {out_dir}/surface_overlap_comparison.png")

print("Done.")