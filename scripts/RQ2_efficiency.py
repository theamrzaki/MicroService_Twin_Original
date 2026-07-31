import pandas as pd
import numpy as np

# 1. Load the benchmark dataset
path = 'result_journal/result_RQ1_benchmark_MemOptimised_lowshow_simplegraph.csv'
path_smaller = 'result_journal/result_RQ1_benchmark_MemOptimised_lowshow_simplegraph (Raspberry Pi Smaller).csv'
path_larger = 'result_journal/result_RQ1_benchmark_memOptimized_lowshow_simplegraph  (Raspberrt Pi Larger).csv'

df = pd.read_csv(path)
df = df[~df['FREQ_DOMAIN'].isin(['Eadro', 'AnoFusion'])]
path2 = 'result_journal/result_RQ1_benchmark_MemOptimised_lowshow_simplegraph EadroAnoFusion.csv'
df2 = pd.read_csv(path2)
df = pd.concat([df, df2], ignore_index=True)
df['device'] = 'server'

# remove "FITS_Legendre" from the dataframe where device is server
df_smaller = pd.read_csv(path_smaller)
df_smaller = df_smaller[~df_smaller['FREQ_DOMAIN'].isin(['Eadro', 'AnoFusion'])]
path2_smaller = 'result_journal/result_RQ1_benchmark_MemOptimised_lowshow_simplegraph (Raspberry Pi Smaller) EadroAnoFusionMedicine.csv'
df2_smaller = pd.read_csv(path2_smaller)
df_smaller = pd.concat([df_smaller, df2_smaller], ignore_index=True)
df_smaller['device'] = 'raspberry_pi_smaller'


df_larger = pd.read_csv(path_larger)
df_larger = df_larger[~df_larger['FREQ_DOMAIN'].isin(['Eadro', 'AnoFusion'])]
path2_larger = 'result_journal/result_RQ1_benchmark_memOptimized_lowshow_simplegraph  (Raspberrt Pi Larger) EadroAnoFusionMedicine.csv'
df2_larger = pd.read_csv(path2_larger)
df_larger = pd.concat([df_larger, df2_larger], ignore_index=True)
df_larger['device'] = 'raspberry_pi_larger'

df = pd.concat([df, df_larger ,df_smaller], ignore_index=True)
df['FREQ_DOMAIN'] = df['FREQ_DOMAIN'].replace({'encoder_decoder': 'MSTGAD', 'FITS_Legendre': 'OrEdge'})
# remove Medicine with SN with params 148308 #as it worked with the huge feature space, that didnt work with TT, so the exps were repeated with the smaller feature space
df = df[~((df['FREQ_DOMAIN'] == 'Medicine') & (df['datasource'] == 'SN') & (df['total_params'] == 148308))]

# 2. Define our target datasets (columns) and model variants (rows)
datasets = ['SN', 'TT', 'MSDS']
variants = ['Eadro', 'AnoFusion', 'MSTGAD', 'Art', 'Medicine', 'OrEdge']

# 3. Define mapping for Server (GPU/CPU) and Raspberry Pi metrics
server_metrics = [
    #("Inference Time (GPU) [ms]", "(benchmark_result) inference_time_ms", 2),
    ("Training Time (GPU) [s]", "training_time_per_epoch", 1),
    #("Peak Mem (GPU) [MB]", "(benchmark_result) inference_memory_mb", 1),
    #("Energy Usage (GPU) [J]", "energy_per_sample_joules (GPU)", 3),
    ("FLOPs [M]", "(benchmark_result) flops_million", 1),
    ("Reconstuction Params", "reconstruction_params", 1),
]

raspberry_metrics = [
    ("Inference Time (CPU) [ms]", "(benchmark_result) inference_time_ms", 2),
     ("Max Mem (CPU) [MB]", "peak_memory_mb (CPU)", 1),
    #("Total Mem (CPU) [MB]", "(benchmark_result) inference_memory_mb", 1),
    #("Energy Usage (CPU) [J]", "energy_per_sample_joules (CPU)", 3)
]

def get_efficiency_color_macro(val, min_val, max_val):
    """
    Maps efficiency means along a clear Green -> Yellow -> Red gradient.
    Lower is better: min_val gets green, max_val gets red.
    """
    if pd.isna(val) or min_val == max_val or np.isnan(min_val) or np.isnan(max_val):
        return ""
    
    # Normalize value between 0 (min/best) and 1 (max/worst)
    norm = (val - min_val) / (max_val - min_val)
    max_opacity = 35  # Consistently uses your 35% vibrant rendering setting
    
    if norm < 0.5:
        # Scale from 0% yellow (pure green) to 100% yellow at the midpoint
        pct_yellow = int(norm * 2 * 100)
        return f"\\cellcolor{{yellow!{pct_yellow}!green!{max_opacity}}}"
    else:
        # Scale from 0% red (pure yellow) to 100% red at the maximum
        pct_red = int((norm - 0.5) * 2 * 100)
        return f"\\cellcolor{{red!{pct_red}!yellow!{max_opacity}}}"

# Helper function to generate table rows for a device category
def generate_device_rows(device_name, hardware_spec, metrics_list, device_slug):
    latex_lines = []
    first_device_line = True
    
    for idx, (metric_label, csv_col, precision) in enumerate(metrics_list):
        first_metric_line = True
        
        for variant in variants:
            if first_device_line:
                dev_col = f"\\smash{{\\begin{{tabular}}[t]{{l}}\\textbf{{{device_name}}}\\\\ {hardware_spec}\\end{{tabular}}}}"
            else:
                dev_col = ""
            
            metric_col = metric_label if first_metric_line else ""
            
            row_values = []
            for ds in datasets:
                # 1. Gather all means for this specific metric + dataset to find min/max baseline bounds
                means_for_bounds = [
                    df[(df['FREQ_DOMAIN'] == v) & (df['datasource'] == ds) & (df['device'] == device_slug)][csv_col].mean()
                    for v in variants
                ]
                means_for_bounds = [m for m in means_for_bounds if not pd.isna(m)]
                
                min_val = min(means_for_bounds) if means_for_bounds else np.nan
                max_val = max(means_for_bounds) if means_for_bounds else np.nan
                
                # 2. Extract run metrics for the target cell
                cell_subset = df[
                    (df['FREQ_DOMAIN'] == variant) & 
                    (df['datasource'] == ds) & 
                    (df['device'] == device_slug)
                ][csv_col]
                
                if not cell_subset.empty and not pd.isna(cell_subset.mean()):
                    mean_val = cell_subset.mean()
                    std_val = cell_subset.std()
                    
                    color_prefix = get_efficiency_color_macro(mean_val, min_val, max_val)
                    
                    # 3. Format output string with trailing ± deviation elements
                    if not pd.isna(std_val) and std_val > 0:
                        cell_text = f"{color_prefix}{mean_val:.{precision}f} {{\\scriptsize$\\pm${std_val:.{precision}f}}}"
                    else:
                        cell_text = f"{color_prefix}{mean_val:.{precision}f} {{\\scriptsize$\\pm$0.000}}"
                    row_values.append(cell_text)
                else:
                    row_values.append("-")
            
            clean_variant = variant.replace('_', '\\_')
            values_str = " & ".join(row_values)
            latex_lines.append(f"{dev_col:<35} & {metric_col:<35} & {clean_variant:<20} & {values_str} \\\\")
            
            first_device_line = False
            first_metric_line = False
            
        if idx < len(metrics_list) - 1:
            latex_lines.append("\\cmidrule{2-6}")
            
    return latex_lines

# 4. Assemble the complete LaTeX document structure
latex_table = []
latex_table.append("\\begin{table*}[t]")
latex_table.append("\\caption{Efficiency and Resource Consumption Analysis Across Devices (Mean $\\pm$ SD)}")
latex_table.append("\\label{tab:efficiency_metrics}")
latex_table.append("\\centering")
latex_table.append("\\scriptsize")
latex_table.append("\\begin{tabular}{llcccc}")
latex_table.append("\\toprule")
latex_table.append("\\textbf{Device} & \\textbf{Metric} & \\textbf{Model Variant} & \\textbf{SN} & \\textbf{TT} & \\textbf{MSDS} \\\\")
latex_table.append("\\midrule")

latex_table.extend(generate_device_rows("Server", "\\footnotesize (GPU: RTX 3070 CPU:i9)", server_metrics, "server"))
latex_table.append("\\midrule[1.5pt]")
latex_table.extend(generate_device_rows("Raspberry Pi 5", "\\footnotesize (ARM Cortex-A76 16GB-Ram)", raspberry_metrics, "raspberry_pi_larger"))
latex_table.append("\\midrule[1.5pt]")
latex_table.extend(generate_device_rows("Raspberry Pi 3", "\\footnotesize (ARM Cortex-A53 1GB-Ram)", raspberry_metrics, "raspberry_pi_smaller"))

latex_table.append("\\bottomrule")
latex_table.append("\\end{tabular}")
latex_table.append("\\end{table*}")

print("\n".join(latex_table))