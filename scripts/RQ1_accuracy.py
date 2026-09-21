import pandas as pd
import numpy as np
import re
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

# 1. Load the benchmark dataset
path = 'result_journal/result_RQ1_benchmark_MemOptimised_lowshow_simplegraph.csv'
df = pd.read_csv(path)
#remove Eadro and AnoFusion from the dataset
df = df[~df['FREQ_DOMAIN'].isin(['Eadro', 'AnoFusion'])]

path2 = 'result_journal/result_RQ1_benchmark_MemOptimised_lowshow_simplegraph EadroAnoFusion.csv'
df2 = pd.read_csv(path2)
# Concatenate the two dataframes
df = pd.concat([df, df2], ignore_index=True)
# remove Medicine with SN with params 148308 #as it worked with the huge feature space, that didnt work with TT, so the exps were repeated with the smaller feature space
#i have removed these lines 
#print(f"number of rows before filtering: {len(df)}")
#df = df[~((df['FREQ_DOMAIN'] == 'Medicine') & (df['datasource'] == 'SN') & (df['total_params'] == 148308))]
#print(f"number of rows after filtering: {len(df)}")
#rename encoder_decoder to MSTGAD
#rename FITS_Legendre to OrEdge
df['FREQ_DOMAIN'] = df['FREQ_DOMAIN'].replace({'encoder_decoder': 'MSTGAD', 'FITS_Legendre': 'OrEdge'})

def parse_info_dict(info_str):
    """Parses accuracy markers safely from text telemetry logs."""
    if pd.isna(info_str):
        return {}
    matches = re.findall(r'([a-zA-Z0-9_]+)\s*:\s*([0-9.]+)', info_str)
    return {k: float(v) for k, v in matches}

# Extract metrics from info_dict columns
parsed_metrics = df['info_dict'].apply(parse_info_dict)
metrics_df = pd.json_normalize(parsed_metrics)
df = pd.concat([df, metrics_df], axis=1)

# 2. Define target parameters
datasets = ['MSDS','SN', 'TT']
variants = ['Eadro', 'AnoFusion', 'MSTGAD', 'Art', 'Medicine','DeepHunt', 'OrEdge']

accuracy_metrics = [
    ("Precision", "pr", 3),
    ("Recall", "rc", 3),
    ("Avg Precision", "ap", 3),
    ("F1-Score", "f1", 3),
    ("AUC-ROC", "auc", 3)
]

def get_accuracy_color_macro(val, min_val, max_val):
    """Generates continuous 35% vibrant background color scaling for accuracy mean metrics."""
    if pd.isna(val) or min_val == max_val or np.isnan(min_val) or np.isnan(max_val):
        return ""
    
    norm = (val - min_val) / (max_val - min_val)
    max_opacity = 35 
    
    if norm > 0.5:
        factor = (norm - 0.5) * 2
        pct_green = int(factor * 100)
        return f"\\cellcolor{{green!{pct_green}!yellow!{max_opacity}}}"
    else:
        factor = norm * 2
        pct_yellow = int(factor * 100)
        pct_red = 100 - pct_yellow
        return f"\\cellcolor{{yellow!{pct_yellow}!red!{max_opacity}}}"

# 3. Compute structural groupings (Mean and Std Dev) per Model & Dataset combination
# Filter to server instances where complete evaluation runs take place
server_df = df

# 4. Generate rows
latex_lines = []
for idx, (metric_label, csv_col, precision) in enumerate(accuracy_metrics):
    first_metric_line = True
    
    for variant in variants:
        metric_col = metric_label if first_metric_line else ""
        row_values = []
        
        for ds in datasets:
            # Find the max/min of the MEAN scores across variants to establish bounds for this dataset
            means_for_bounds = [
                server_df[(server_df['FREQ_DOMAIN'] == v) & (server_df['datasource'] == ds)][csv_col].mean()
                for v in variants
            ]
            means_for_bounds = [m for m in means_for_bounds if not pd.isna(m)]
            
            min_val = min(means_for_bounds) if means_for_bounds else np.nan
            max_val = max(means_for_bounds) if means_for_bounds else np.nan
            
            # Extract data subset for this exact cell
            cell_subset = server_df[(server_df['FREQ_DOMAIN'] == variant) & (server_df['datasource'] == ds)][csv_col]
            
            if not cell_subset.empty and not pd.isna(cell_subset.mean()):
                mean_val = cell_subset.mean()
                std_val = cell_subset.std()
                
                color_prefix = get_accuracy_color_macro(mean_val, min_val, max_val)
                
                # Format text: Mean values followed by subscript/small standard deviations if available
                if not pd.isna(std_val) and std_val > 0:
                    cell_text = f"{color_prefix}{mean_val:.{precision}f} {{\\scriptsize$\\pm${std_val:.{precision}f}}}"
                else:
                    # Fallback for combinations with only 1 seed run (e.g., FITS_Legendre)
                    cell_text = f"{color_prefix}{mean_val:.{precision}f} {{\\scriptsize$\\pm$0.000}}"
                row_values.append(cell_text)
            else:
                row_values.append("-")
                
        clean_variant = variant.replace('_', '\\_')
        values_str = " & ".join(row_values)
        latex_lines.append(f"{metric_col:<25} & {clean_variant:<20} & {values_str} \\\\")
        first_metric_line = False
        
    if idx < len(accuracy_metrics) - 1:
        latex_lines.append("\\cmidrule{1-5}")

# 5. Assemble final LaTeX wrapped structure
latex_table = []
latex_table.append("\\begin{table}[t]")
latex_table.append("\\caption{Anomaly Detection Accuracy Performance Across Datasets (Mean $\\pm$ SD)}")
latex_table.append("\\label{tab:accuracy_metrics}")
latex_table.append("\\centering")
latex_table.append("\\scriptsize")
latex_table.append("\\begin{tabular}{p{1.5cm}p{1.3cm}p{1.4cm}p{1.4cm}p{1.4cm}}")
latex_table.append("\\toprule")
latex_table.append("\\textbf{Metric} & \\textbf{Model Variant} & \\textbf{MSDS} & \\textbf{SN} & \\textbf{TT} \\\\")
latex_table.append("\\midrule")
latex_table.extend(latex_lines)
latex_table.append("\\bottomrule")
latex_table.append("\\end{tabular}")
latex_table.append("\\end{table}")

#print("\n".join(latex_table))

with open('sections/Results/RQ1_accuracy.tex', 'w') as f:
    f.write("\n".join(latex_table))
print("LaTeX table saved to sections/Results/RQ1_accuracy.tex !!")


# ============================================================
# Statistical Significance Analysis: OrEdge vs. Best Baseline
# ============================================================

# Comparisons selected based on the best F1-performing baseline
# for each dataset in the main RQ1 results.
#comparisons = {
#    'SN': 'Art',
#    'TT': 'Art',
#    'MSDS': 'MSTGAD'
#}
# get max F1 baseline for each dataset
comparisons = {}
for dataset in datasets:
    baseline_df = df[
        (df['datasource'] == dataset) &
        (df['FREQ_DOMAIN'] != 'OrEdge')
    ]
    if not baseline_df.empty:
        best_baseline_row = baseline_df.loc[baseline_df['f1'].idxmax()]
        comparisons[dataset] = best_baseline_row['FREQ_DOMAIN']
    else:
        comparisons[dataset] = None

stat_results = []

for dataset, baseline in comparisons.items():

    # --------------------------------------------------------
    # Select OrEdge and baseline runs
    # --------------------------------------------------------
    or_edge = df[
        (df['datasource'] == dataset) &
        (df['FREQ_DOMAIN'] == 'OrEdge')
    ][['random_seed', 'f1']].copy()

    baseline_df = df[
        (df['datasource'] == dataset) &
        (df['FREQ_DOMAIN'] == baseline)
    ][['random_seed', 'f1']].copy()

    or_edge = or_edge.rename(
        columns={'f1': 'f1_or_edge'}
    )

    baseline_df = baseline_df.rename(
        columns={'f1': 'f1_baseline'}
    )

    # --------------------------------------------------------
    # Pair runs using the same random seed
    # --------------------------------------------------------
    paired = pd.merge(
        or_edge,
        baseline_df,
        on='random_seed',
        how='inner'
    ).dropna()

    print(f"\n===== {dataset}: OrEdge vs {baseline} =====")
    print(paired)

    if len(paired) < 2:
        print("Not enough paired runs for statistical testing.")
        continue

    # --------------------------------------------------------
    # Paired differences
    # --------------------------------------------------------
    differences = (
        paired['f1_or_edge'] -
        paired['f1_baseline']
    )

    # Remove exact zero differences for effect-size calculation
    nonzero_diff = differences[differences != 0]

    # --------------------------------------------------------
    # Wilcoxon signed-rank test
    # --------------------------------------------------------
    try:
        statistic, p_value = wilcoxon(
            paired['f1_or_edge'],
            paired['f1_baseline'],
            alternative='two-sided'
        )
    except ValueError as e:
        print(f"Wilcoxon test failed: {e}")
        continue

    # --------------------------------------------------------
    # Rank-biserial correlation
    #
    # r_rb = (W+ - W-) / (W+ + W-)
    #
    # Positive values indicate larger OrEdge scores.
    # Negative values indicate larger baseline scores.
    # --------------------------------------------------------
    if len(nonzero_diff) > 0:
        ranks = nonzero_diff.abs().rank(method='average')

        positive_rank_sum = ranks[nonzero_diff > 0].sum()
        negative_rank_sum = ranks[nonzero_diff < 0].sum()

        rank_biserial = (
            positive_rank_sum - negative_rank_sum
        ) / (
            positive_rank_sum + negative_rank_sum
        )
    else:
        rank_biserial = 0.0

    # --------------------------------------------------------
    # Descriptive statistics
    # --------------------------------------------------------
    mean_difference = differences.mean()
    median_difference = differences.median()

    stat_results.append({
        'Dataset': dataset,
        'Baseline': baseline,
        'N': len(paired),

        'OrEdge_F1': paired['f1_or_edge'].mean(),
        'Baseline_F1': paired['f1_baseline'].mean(),

        'Mean_Difference': mean_difference,
        'Median_Difference': median_difference,

        'Wilcoxon_statistic': statistic,
        'p_value': p_value,

        'Rank_Biserial': rank_biserial
    })


# ============================================================
# Multiple-comparison correction
# ============================================================

results_df = pd.DataFrame(stat_results)

if not results_df.empty:

    # Holm correction across the three dataset comparisons
    reject, p_corrected, _, _ = multipletests(
        results_df['p_value'],
        method='holm',
        alpha=0.05
    )

    results_df['p_value_holm'] = p_corrected
    results_df['significant'] = reject

    # --------------------------------------------------------
    # Print statistical results
    # --------------------------------------------------------
    print("\n\n===== Statistical Analysis Results =====")

    print(
        results_df[
            [
                'Dataset',
                'Baseline',
                'N',
                'OrEdge_F1',
                'Baseline_F1',
                'Mean_Difference',
                'Median_Difference',
                'p_value',
                'p_value_holm',
                'Rank_Biserial',
                'significant'
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Generate LaTeX statistical table
    # --------------------------------------------------------
    latex_stat_lines = []

    for _, row in results_df.iterrows():

        dataset = row['Dataset']
        baseline = row['Baseline']
        n = int(row['N'])

        or_edge_mean = row['OrEdge_F1']
        baseline_mean = row['Baseline_F1']
        mean_diff = row['Mean_Difference']
        p_holm = row['p_value_holm']
        effect = row['Rank_Biserial']

        # Significance marker
        if p_holm < 0.001:
            p_text = "$<0.001$"
        else:
            p_text = f"${p_holm:.3f}$"

        latex_stat_lines.append(
            f"{dataset} & {baseline} & {n} & "
            f"{or_edge_mean:.3f} & "
            f"{baseline_mean:.3f} & "
            f"{mean_diff:+.3f} & "
            f"{p_text} & "
            f"{effect:+.3f} \\\\"
        )

    latex_stat_table = []

    latex_stat_table.append("\\begin{table}[t]")
    latex_stat_table.append(
        "\\caption{Statistical comparison of OrEdge and the strongest "
        "F1-performing baseline across datasets.}"
    )
    latex_stat_table.append("\\label{tab:rq1_significance}")
    latex_stat_table.append("\\centering")
    latex_stat_table.append("\\scriptsize")
    latex_stat_table.append(
        "\\begin{tabular}{lccccccc}"
    )
    latex_stat_table.append("\\toprule")
    latex_stat_table.append(
        "\\textbf{Dataset} & "
        "\\textbf{Baseline} & "
        "$n$ & "
        "\\textbf{OrEdge} & "
        "\\textbf{Baseline} & "
        "$\\Delta$F1 & "
        "$p_{\\mathrm{Holm}}$ & "
        "$r_{\\mathrm{rb}}$ \\\\"
    )
    latex_stat_table.append("\\midrule")
    latex_stat_table.extend(latex_stat_lines)
    latex_stat_table.append("\\bottomrule")
    latex_stat_table.append("\\end{tabular}")
    latex_stat_table.append("\\end{table}")

    with open(
        'sections/Results/RQ1_significance.tex',
        'w'
    ) as f:
        f.write("\n".join(latex_stat_table))

    print(
        "\nLaTeX statistical table saved to "
        "sections/Results/RQ1_significance.tex !!"
    )


#across datasources, FREQ_DOMAIN (check if it has 3 seeds using "random_seed" column)
# print missing combinations with the found seeds 
missing_combinations = []
for ds in df["datasource"].unique():
    for arch in df["FREQ_DOMAIN"].unique():
            seeds = df[
                (df["datasource"] == ds) &
                (df["FREQ_DOMAIN"] == arch)
            ]["random_seed"].unique()
            if len(seeds) < 3:
                missing_combinations.append((ds, arch, seeds))
for ds, arch, seeds in missing_combinations:
    print(f"Missing combination: datasource={ds}, arch={arch}, found seeds={seeds}")
if len(missing_combinations) == 0:
    print("All combinations have at least 3 seeds.")