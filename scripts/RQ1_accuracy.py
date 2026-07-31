import pandas as pd
import numpy as np
import re

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
df = df[~((df['FREQ_DOMAIN'] == 'Medicine') & (df['datasource'] == 'SN') & (df['total_params'] == 148308))]

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
datasets = ['SN', 'TT', 'MSDS']
variants = ['Eadro', 'AnoFusion', 'MSTGAD', 'Art', 'Medicine', 'OrEdge']

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
latex_table.append("\\textbf{Metric} & \\textbf{Model Variant} & \\textbf{SN} & \\textbf{TT} & \\textbf{MSDS} \\\\")
latex_table.append("\\midrule")
latex_table.extend(latex_lines)
latex_table.append("\\bottomrule")
latex_table.append("\\end{tabular}")
latex_table.append("\\end{table}")

print("\n".join(latex_table))