import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. LOAD DATASETS
Dataset_Name = "MSTGAD"
with open(f'result_journal\case_study_{Dataset_Name}\({Dataset_Name}) case_output_{Dataset_Name}_300.json', 'r') as f:
    art_data = json.load(f)
with open(f'result_journal\case_study_{Dataset_Name}\({Dataset_Name}) case_output_FITS_Legendre_300.json', 'r') as f:
    fits_data = json.load(f)


def get_stats(data):
    tp = len(data['cases'].get('TP', []))
    fn = len(data['cases'].get('FN', []))
    return tp + fn, tp, fn

art_t, art_tp, art_fn = get_stats(art_data)
fits_t, fits_tp, fits_fn = get_stats(fits_data)

# LaTeX Generation
print(f"""
\\begin{{table}}[h]
\\centering
\\caption{{Case Study Detection Distribution (N=20)}}
\\label{{tab:case_dist}}
\\begin{{tabular}}{{@{{}}lccc@{{}}}}
\\toprule
\\textbf{{Model}} & \\textbf{{Total Cases}} & \\textbf{{Detected (TP)}} & \\textbf{{Missed (FN)}} \\\\ \\midrule
Baseline (Art) & {art_t} & {art_tp} & {art_fn} \\\\
Proposed (FITS) & {fits_t} & {fits_tp} & {fits_fn} \\\\ \\bottomrule
\\end{{tabular}}
\\end{{table}}
""")



# 2. HELPER FUNCTIONS

def get_behavioral_df(data):
    """Converts JSON case data into a DataFrame for behavioral mapping."""
    results = []
    for cat in ['TP', 'FP', 'FN']:
        for case in data['cases'].get(cat, []):
            log_count = sum(sum(w['count'] for w in window) for window in case['log'])
            results.append({
                'id': case['id'],
                'Category': cat,
                'Score': case['score'],
                'Log Count': log_count,
                'metric': case['metric']
            })
    return pd.DataFrame(results)

# Prepare DataFrames and identify comparison cases
df_art = get_behavioral_df(art_data)
df_fits = get_behavioral_df(fits_data)

# --- FIGURE 5: BEHAVIORAL MAPS (Labeled Scatter Plots) ---
# 2. CALCULATE GLOBAL STATISTICS (FOR Z-SCORE NORMALIZATION)
def get_global_stats(data_list):
    m_vals, l_vals, t_vals = [], [], []
    for data in data_list:
        for cat in ['TP', 'FP', 'FN']:
            for c in data['cases'].get(cat, []):
                m_vals.append(np.mean(c['metric']))
                l_vals.append(sum(sum(w['count'] for w in win) for win in c['log']))
                t_vals.append(np.mean(c['trace']))
    
    return {
        'm': (np.mean(m_vals), np.std(m_vals)),
        'l': (np.mean(l_vals), np.std(l_vals)),
        't': (np.mean(t_vals), np.std(t_vals))
    }

stats = get_global_stats([art_data, fits_data])

# 3. PREPARE DATASET WITH Z-SCORE DOMINANCE
def prepare_zscore_df(data, stats):
    results = []
    for cat in ['TP', 'FP', 'FN']:
        for case in data['cases'].get(cat, []):
            m_val = np.mean(case['metric'])
            l_val = sum(sum(w['count'] for w in window) for window in case['log'])
            t_val = np.mean(case['trace'])
            
            # Calculate Z-Scores
            z_m = (m_val - stats['m'][0]) / stats['m'][1]
            z_l = (l_val - stats['l'][0]) / stats['l'][1]
            z_t = (t_val - stats['t'][0]) / stats['t'][1]
            
            # Determine which modality is "most anomalous" relative to its own norm
            z_scores = {'Metrics': z_m, 'Logs': z_l, 'Traces': z_t}
            dominant = max(z_scores, key=z_scores.get)
            
            results.append({
                'id': case['id'],
                'Category': cat,
                'Score': case['score'],
                'Log Count': l_val,
                'Dominant': dominant,
                'Z_Max': z_scores[dominant]
            })
    return pd.DataFrame(results)

df_art = prepare_zscore_df(art_data, stats)
df_fits = prepare_zscore_df(fits_data, stats)


def get_automated_ids(df_art, df_fits, top_n=5):
    """
    Automates the selection of IDs for the case study.
    1. Aligned TPs: Highest scores where BOTH models agreed.
    2. FITS Victories: Cases where Art was FN but FITS was TP.
    """
    # Merge to compare side-by-side
    merged = pd.merge(df_art, df_fits, on='id', suffixes=('_Art', '_FITS'))
    
    # 1. Top Aligned TPs (Common successes)
    aligned_tp = merged[(merged['Category_Art'] == 'TP') & 
                        (merged['Category_FITS'] == 'TP')].nlargest(top_n, 'Score_FITS')
    
    # 2. FITS Victories (Misaligned TPs - The 'Legendre Win')
    fits_wins = merged[(merged['Category_Art'] == 'FN') & 
                       (merged['Category_FITS'] == 'TP')].nlargest(top_n, 'Score_FITS')
    
    art_wins = merged[(merged['Category_Art'] == 'TP') & 
                       (merged['Category_FITS'] == 'FN')]
    return {
        'aligned': aligned_tp['id'].tolist(),
        'fits_wins': fits_wins['id'].tolist(),
        'art_wins': art_wins['id'].tolist()
    }

# Execute automation
selected_ids = get_automated_ids(df_art, df_fits)
print(f"Automated Case Study IDs: {selected_ids}")




# 1. Load Data
with open(f'result_journal\case_study_{Dataset_Name}\({Dataset_Name}) case_output_FITS_Legendre_300.json', 'r') as f:
    fits_data = json.load(f)

def get_case_obj(data, cid):
    for cat in ['TP', 'FP', 'FN']:
        for c in data['cases'].get(cat, []):
            if c["id"] == cid: return c
    return None

import os
import pandas as pd
import numpy as np
import os

# Create base directory for transparency and debugging
base_dir = 'figs/results_casestudy'
os.makedirs(base_dir, exist_ok=True)

def save_case_row_with_data(case_id, category_label, filename):
    # Retrieve objects
    c_fits = get_case_obj(fits_data, case_id)
    c_art = get_case_obj(art_data, case_id)
    
    # Create sub-directory for this specific case's raw data
    case_data_dir = os.path.join(base_dir, f"data_{case_id}")
    os.makedirs(case_data_dir, exist_ok=True)

    # --- 1. MODALITY DATA EXTRACTION ---
    
    # A. Metrics Data (Force to 1D)
    metrics_series = np.concatenate(c_fits['metric'], axis=0).mean(axis=1).flatten()
    pd.DataFrame({
        'timestep': np.arange(len(metrics_series)), 
        'value': metrics_series
    }).to_csv(os.path.join(case_data_dir, f"{case_id}_metrics_raw.csv"), index=False)

    # B. Traces Data (Force to 1D)
    traces_series = np.concatenate(c_fits['trace'], axis=0).mean(axis=1).flatten()
    pd.DataFrame({
        'timestep': np.arange(len(traces_series)), 
        'value': traces_series
    }).to_csv(os.path.join(case_data_dir, f"{case_id}_traces_raw.csv"), index=False)

    # C. Logs Data (Template Distribution)
    log_summary = {}
    for win in c_fits['log']:
        for ent in win:
            tid = f"T{ent['template_id']}"
            log_summary[tid] = log_summary.get(tid, 0) + ent['count']
    
    pd.DataFrame(list(log_summary.items()), columns=['Template_ID', 'Count']).sort_values(
        by='Count', ascending=False).to_csv(
        os.path.join(case_data_dir, f"{case_id}_logs_dist.csv"), index=False
    )

    # D. Verdict Data (The Numerical Proof)
    verdict_data = {
        'Model': ['Baseline (Art)', 'Proposed (FITS)'],
        'Score': [c_art['score'], c_fits['score']],
        'Threshold': [art_data['threshold'], fits_data['threshold']],
        'Detection_Ratio': [c_art['score']/art_data['threshold'], c_fits['score']/fits_data['threshold']],
        'Result': [c_art.get('category', 'N/A'), c_fits.get('category', 'N/A')]
    }
    pd.DataFrame(verdict_data).to_csv(
        os.path.join(case_data_dir, f"{case_id}_verdict.csv"), index=False
    )

    # --- 2. FIGURE GENERATION (Unchanged Layout) ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 3), gridspec_kw={'width_ratios': [1, 1, 1]})
    
    axes[0].plot(metrics_series, color='#3498db', linewidth=1.5)
    axes[0].set_title(f"Metrics ({case_id})", fontweight='bold')
    
    axes[1].plot(traces_series, color='#e67e22', linewidth=1.5)
    axes[1].set_title(f"Traces ({case_id})", fontweight='bold')
    
    axes[2].bar(log_summary.keys(), log_summary.values(), color='#9b59b6', alpha=0.7)
    axes[2].set_title(f"Logs ({case_id})", fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=7)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# --- 3. EXECUTION ---
for i, cid in enumerate(selected_ids['fits_wins'][:2]):
    save_case_row_with_data(cid, "FITS Victory", f"{base_dir}/({Dataset_Name}) row_fits_{i}.pdf")

for i, cid in enumerate(selected_ids['art_wins'][:2]):
    save_case_row_with_data(cid, "Art Victory", f"{base_dir}/({Dataset_Name}) row_art_{i}.pdf")

for i, cid in enumerate(selected_ids['aligned'][:2]):
    save_case_row_with_data(cid, "Aligned TP", f"{base_dir}/({Dataset_Name}) row_aligned_{i}.pdf")