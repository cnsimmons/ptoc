#!/usr/bin/env python3
### FINAL FOR REVISION ###
# Standalone script for FC, PPI, and partial-correlation visualization (Pearson's r)
# Headless-safe (Agg): run from a terminal / srun session; saves PNGs to output_dir.

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter

# Define paths
results_dir = '/user_data/csimmon2/git_repos/ptoc/results'
output_dir = f'{results_dir}/connectivity_comparison'
os.makedirs(output_dir, exist_ok=True)

# Helper functions
def map_to_anatomical_lobe(roi_name):
    """Map ROI names to anatomical lobes"""
    if isinstance(roi_name, bytes):
        roi_name = roi_name.decode('utf-8')
    
    if 'Wang_pIPS' in roi_name:
        return 'Parietal'
    elif 'Wang_LO' in roi_name:
        return 'Temporal'
    
    if 'Vis' in roi_name:
        return 'Occipital'
    elif 'SomMot' in roi_name:
        return 'Somatomotor'
    elif 'DorsAttn' in roi_name:
        if 'Par' in roi_name or 'IPL' in roi_name or 'IPS' in roi_name:
            return 'Parietal'
        elif 'Temp' in roi_name or 'MT' in roi_name:
            return 'Temporal'
        else:
            return 'Parietal'
    elif 'SalVentAttn' in roi_name:
        if 'Ins' in roi_name:
            return 'Insular'
        elif 'Cing' in roi_name or 'ACC' in roi_name:
            return 'Cingulate'
        elif 'Temp' in roi_name:
            return 'Temporal'
        elif 'Par' in roi_name:
            return 'Parietal'
        else:
            return 'Frontal'
    elif 'Limbic' in roi_name:
        if 'Temp' in roi_name:
            return 'Temporal'
        else:
            return 'Frontal'
    elif 'Cont' in roi_name:
        if 'Par' in roi_name or 'IPL' in roi_name or 'IPS' in roi_name:
            return 'Parietal'
        elif 'Temp' in roi_name or 'MT' in roi_name:
            return 'Temporal'
        elif 'Cing' in roi_name:
            return 'Cingulate'
        else:
            return 'Frontal'
    elif 'Default' in roi_name:
        if 'Par' in roi_name:
            return 'Parietal'
        elif 'Temp' in roi_name:
            return 'Temporal'
        elif 'PCC' in roi_name or 'Cing' in roi_name:
            return 'Cingulate'
        else:
            return 'Frontal'
    else:
        return 'Other'

def clean_roi_name(roi_name):
    """Clean up ROI names for better labeling"""
    if isinstance(roi_name, bytes):
        roi_name = roi_name.decode('utf-8')
    
    if 'Wang_' in roi_name:
        return roi_name.replace('Wang_', '')
    
    cleaned = roi_name.replace('7Networks_', '')
    
    parts = cleaned.split('_')
    if len(parts) > 2:
        return parts[-2] + '_' + parts[-1]
    elif len(parts) > 1:
        return parts[-1]
    else:
        return cleaned

def get_hemisphere_and_region(roi_name):
    """Determine hemisphere and base region for organizing pairs"""
    if isinstance(roi_name, bytes):
        roi_name = roi_name.decode('utf-8')
    
    if 'LH' in roi_name:
        hemisphere = 'L'
    elif 'RH' in roi_name:
        hemisphere = 'R'
    else:
        if 'Wang_pIPS' in roi_name or 'Wang_LO' in roi_name:
            hemisphere = 'X'
        else:
            hemisphere = 'X'
    
    cleaned = roi_name.replace('7Networks_', '')
    cleaned = cleaned.replace('LH_', '').replace('RH_', '')
    
    if 'Wang_' in cleaned:
        base_region = cleaned.replace('Wang_', '')
    else:
        base_region = re.sub(r'_\d+$', '', cleaned)
    
    return hemisphere, base_region

def load_data_for_visualization(analysis_type='fc'):
    """Load data from CSV and prepare it for visualization"""
    csv_path = f'{output_dir}/bilateral_{analysis_type}_connectivity_fingerprint_results_pearsonr.csv'
    
    if not os.path.exists(csv_path):
        print(f"Error: {analysis_type.upper()} results file not found at {csv_path}")
        return None, None
        
    results_df = pd.read_csv(csv_path)
    print(f"Loaded {analysis_type.upper()} data with {len(results_df)} ROIs")
    
    roi_data = {
        'mean_pips': results_df['pIPS_Connectivity'].values,
        'mean_lo': results_df['LO_Connectivity'].values,
        'diff_profile': results_df['Difference'].values,
        'ci_lower': results_df['CI_Lower'].values if 'CI_Lower' in results_df else None,
        'ci_upper': results_df['CI_Upper'].values if 'CI_Upper' in results_df else None,
        'sig_combined': results_df['Combined_Significant'].values
    }
    
    return results_df, roi_data

def visualize_anatomical_organization(results_df, roi_data, analysis_type='fc'):
    """Create anatomical organization visualization"""
    print(f"\nCreating {analysis_type.upper()} anatomical organization visualization...")
    
    mean_pips = roi_data['mean_pips']
    mean_lo = roi_data['mean_lo']
    diff_profile = roi_data['diff_profile']
    ci_lower = roi_data['ci_lower']
    ci_upper = roi_data['ci_upper']
    sig_combined = roi_data['sig_combined']
    
    results_df['Anatomical_Lobe'] = results_df['ROI_Name'].apply(map_to_anatomical_lobe)
    results_df['Clean_Name'] = results_df['ROI_Name'].apply(clean_roi_name)
    
    hemisphere_region = results_df['ROI_Name'].apply(get_hemisphere_and_region)
    results_df['Hemisphere'] = [h for h, r in hemisphere_region]
    results_df['Base_Region'] = [r for h, r in hemisphere_region]
    
    lobe_order = [
        'Occipital',
        'Parietal',
        'Temporal',
        'Insular',
        'Cingulate',
        'Somatomotor',
        'Frontal',
        'Other'
    ]
    
    lobe_cat = pd.Categorical(results_df['Anatomical_Lobe'], categories=lobe_order, ordered=True)
    results_df['Lobe_Sorted'] = lobe_cat
    
    lobe_colors = {
        'Frontal': '#3498db',
        'Somatomotor': '#f1c40f',
        'Parietal': '#e74c3c',
        'Temporal': '#2ecc71',
        'Occipital': '#9b59b6',
        'Insular': '#f39c12',
        'Cingulate': '#1abc9c',
        'Other': '#7f8c8d'
    }
    
    def custom_sort(row):
        lobe_idx = lobe_order.index(row['Anatomical_Lobe']) if row['Anatomical_Lobe'] in lobe_order else 999
        hemi_idx = 0 if row['Hemisphere'] == 'L' else 1 if row['Hemisphere'] == 'R' else 2
        return (lobe_idx, row['Base_Region'], hemi_idx)
    
    results_df['sort_key'] = results_df.apply(custom_sort, axis=1)
    results_df_sorted = results_df.sort_values('sort_key')
    
    sorted_indices = results_df_sorted.index.values
    
    mean_pips_sorted = results_df_sorted['pIPS_Connectivity'].values
    mean_lo_sorted = results_df_sorted['LO_Connectivity'].values
    diff_profile_sorted = results_df_sorted['Difference'].values
    sig_sorted = results_df_sorted['Combined_Significant'].values
    
    significant_roi_positions = []
    lobe_sections = []
    
    current_lobe = None
    start_idx = 0
    
    for i, idx in enumerate(sorted_indices):
        lobe = results_df_sorted.iloc[i]['Anatomical_Lobe']
        if lobe != current_lobe:
            if current_lobe is not None:
                lobe_sections.append((current_lobe, start_idx, i-1))
            current_lobe = lobe
            start_idx = i
    
    if current_lobe is not None:
        lobe_sections.append((current_lobe, start_idx, len(sorted_indices)-1))
    
    for lobe, start, end in lobe_sections:
        section_indices = sorted_indices[start:end+1]
        section_diff = diff_profile_sorted[start:end+1]
        section_sig = sig_sorted[start:end+1]
        
        sig_positions = np.where(section_sig)[0]
        if len(sig_positions) > 0:
            sig_diffs = section_diff[sig_positions]
            
            if np.max(sig_diffs) > 0:
                max_sig_idx = sig_positions[np.argmax(sig_diffs)]
                significant_roi_positions.append(start + max_sig_idx)
            
            if np.min(sig_diffs) < 0:
                min_sig_idx = sig_positions[np.argmin(sig_diffs)]
                significant_roi_positions.append(start + min_sig_idx)
    
    plt.figure(figsize=(18, 10))
    
    ax1 = plt.subplot(2, 1, 1)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    x = np.arange(len(mean_pips_sorted))
    
    y_min, y_max = min(min(mean_pips_sorted), min(mean_lo_sorted)), max(max(mean_pips_sorted), max(mean_lo_sorted))
    y_range = y_max - y_min
    
    for lobe, start, end in lobe_sections:
        plt.axvspan(start - 0.5, end + 0.5, alpha=0.15, color=lobe_colors[lobe])
        label_y = y_min - 0.1 * y_range
        if lobe == 'Insular':
            lobe_fontsize = 12
        elif lobe == 'Cingulate':
            lobe_fontsize = 12
        else:
            lobe_fontsize = 14
        plt.text((start + end) / 2, label_y, lobe, ha='center', fontsize=lobe_fontsize, fontweight='bold')
    
    for i in range(len(x)):
        plt.plot([x[i], x[i]], [mean_pips_sorted[i], mean_lo_sorted[i]], color='gray', linestyle='-', linewidth=0.5)
    
    plt.scatter(x, mean_pips_sorted, color='#4ac0c0', s=20, label='Dorsal', edgecolors='black', linewidths=0.5)
    plt.scatter(x, mean_lo_sorted, color='#ff9b83', s=20, label='Ventral', edgecolors='black', linewidths=0.5)
    
    plt.ylim(y_min - 0.2 * y_range, y_max + 0.1 * y_range)
    
    plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
    plt.tick_params(axis='both', labelsize=14)
    plt.ylabel("Connectivity (Pearson's r)", fontsize=18)
    plt.legend(loc='upper left', fontsize=14)
    plt.xlim(-0.5, len(mean_pips_sorted) - 0.5)
    
    ax2 = plt.subplot(2, 1, 2)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    diff_min, diff_max = min(diff_profile_sorted), max(diff_profile_sorted)
    diff_range = diff_max - diff_min
    
    base_colors = ['#4ac0c0' if val > 0 else '#ff9b83' for val in diff_profile_sorted]
    bars = plt.bar(x, diff_profile_sorted, color=base_colors)
    
    for i, (bar, is_sig) in enumerate(zip(bars, sig_sorted)):
        if not is_sig:
            bar.set_alpha(0.3)
        else:
            bar.set_edgecolor('black')
            bar.set_linewidth(0.6)
    
    plt.axhline(y=0, color='black', linestyle='-')
    
    sig_legend_elements = [
        Patch(facecolor='gray', edgecolor='black', linewidth=0.6, alpha=1.0, label='Significant'),
        Patch(facecolor='gray', alpha=0.3, label='Non-significant')
    ]
    sig_legend = plt.legend(handles=sig_legend_elements, loc='upper right', fontsize=14)
    
    plt.gca().add_artist(sig_legend)
    
    pref_legend_elements = [
        Patch(facecolor='#4ac0c0', label='Dorsal'),
        Patch(facecolor='#ff9b83', label='Ventral')
    ]
    plt.legend(handles=pref_legend_elements, loc='upper left', fontsize=14)
    
    for lobe, start, end in lobe_sections:
        plt.axvspan(start - 0.5, end + 0.5, alpha=0.15, color=lobe_colors[lobe])
    
    plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
    plt.tick_params(axis='both', labelsize=14)
    
    for position in significant_roi_positions:
        orig_idx = sorted_indices[position]
        roi_name = results_df.loc[orig_idx, 'Clean_Name']
        diff_value = diff_profile_sorted[position]
        
        # Skip specific labels
        if 'Vis_13' in roi_name:
            continue
        
        plt.annotate(f'{roi_name}',
                    xy=(position, diff_value),
                    xytext=(0, 20 if diff_value >= 0 else -25),
                    textcoords='offset points',
                    ha='center',
                    va='bottom' if diff_value >= 0 else 'top',
                    fontsize=11,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', lw=1.5))
    
    plt.ylabel('Dorsal - Ventral Difference', fontsize=18)
    plt.xlabel('ROI ID (Organized by Anatomical Region, L-R Pairs)', fontsize=18)
    plt.xlim(-0.5, len(mean_pips_sorted) - 0.5)
    
    # r-space: data-driven symmetric limits; lock per-figure once eyeballed
    dlim = np.nanmax(np.abs(diff_profile_sorted)) * 1.25
    plt.ylim(-dlim, dlim)
    
    current_ylim = plt.ylim()
    y_range_bottom = current_ylim[1] - current_ylim[0]
    
    for lobe, start, end in lobe_sections:
        label_y = current_ylim[0] - 0.1 * y_range_bottom
        if lobe == 'Insular':
            lobe_fontsize = 12
        elif lobe == 'Cingulate':
            lobe_fontsize = 12
        else:
            lobe_fontsize = 14
        plt.text((start + end) / 2, label_y, lobe, ha='center', fontsize=lobe_fontsize, fontweight='bold')
    
    plt.ylim(current_ylim[0] - 0.15 * y_range_bottom, current_ylim[1])
    
    plt.tight_layout()
    
    fig_path = f'{output_dir}/{analysis_type}_anatomical_organization.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Figure saved to {fig_path}")
    
    print(f"\n{analysis_type.upper()} significant peak and bottom ROIs by anatomical lobe:")
    for lobe, start, end in lobe_sections:
        section_indices = sorted_indices[start:end+1]
        section_df = results_df.loc[section_indices]
        
        sig_group = section_df[section_df['Combined_Significant']]
        
        if len(sig_group) > 0:
            print(f"\n{lobe} Lobe:")
            
            pips_group = sig_group[sig_group['Difference'] > 0]
            if len(pips_group) > 0:
                peak_row = pips_group.loc[pips_group['Difference'].idxmax()]
                print(f"  Peak (pIPS > LO): {peak_row['Clean_Name']}, " +
                      f"Diff = {peak_row['Difference']:.3f}")
            else:
                print("  No significant pIPS-preferring ROIs in this lobe")
            
            lo_group = sig_group[sig_group['Difference'] < 0]
            if len(lo_group) > 0:
                bottom_row = lo_group.loc[lo_group['Difference'].idxmin()]
                print(f"  Bottom (LO > pIPS): {bottom_row['Clean_Name']}, " +
                      f"Diff = {bottom_row['Difference']:.3f}")
            else:
                print("  No significant LO-preferring ROIs in this lobe")

# Run visualizations
if __name__ == "__main__":
    fc_results, fc_data = load_data_for_visualization('fc')
    if fc_results is not None and fc_data is not None:
        visualize_anatomical_organization(fc_results, fc_data, analysis_type='fc')

    ppi_results, ppi_data = load_data_for_visualization('ppi')
    if ppi_results is not None and ppi_data is not None:
        visualize_anatomical_organization(ppi_results, ppi_data, analysis_type='ppi')

    partial_results, partial_data = load_data_for_visualization('partial_correlation')
    if partial_results is not None and partial_data is not None:
        visualize_anatomical_organization(partial_results, partial_data, analysis_type='partial_correlation')
