import pandas as pd
import numpy as np
import re

# 1. File Path Configuration
ablation_path = 'result_journal\\result_RQ2_ablations_withNorm_accMem_updatedFreDF_lowshow_simplegraph.csv'
basis_path = 'result_journal\\result_RQ2_basis_withNorm_accMem_updatedFreDF_lowshow_simplegraph.csv'
architecture_path = 'result_journal\\result_RQ2_architecture_updatedFreDF_lowshow_simplegraph.csv'

OrEdge_path = 'result_journal\\result_RQ1_benchmark_MemOptimised_lowshow_simplegraph.csv'



#-------df_ablation, df_basis, df_architecture --------#
df_ablation = pd.read_csv(ablation_path)
# for SN and TT, remove (1.0, 0.5, "LPF", "linear_attn", True,SN) (1.0, 0.2, "LPF", "linear_attn", True,SN) of rec_lambda, auxi_lambda, filter_used, modules_attn, use_normlin, datasource, as it is already included in df_OrEdge in 2 lines below
df_ablation = df_ablation[~((df_ablation['rec_lambda'] == 1.0) & (df_ablation['auxi_lambda'] == 0.5) & (df_ablation['filter_used'] == "LPF") & (df_ablation['modules_attn'] == "linear_attn") & (df_ablation['use_normlin'] == True) & (df_ablation['datasource'] == "SN"))]
df_ablation = df_ablation[~((df_ablation['rec_lambda'] == 1.0) & (df_ablation['auxi_lambda'] == 0.2) & (df_ablation['filter_used'] == "LPF") & (df_ablation['modules_attn'] == "linear_attn") & (df_ablation['use_normlin'] == True) & (df_ablation['datasource'] == "TT"))]

df_basis = pd.read_csv(basis_path)
df_basis = df_basis[df_basis['basis_type'] != 'legendre']#remove the rows with basis_type == legendre from df_basis (as they are already included in df_OrEdge)

df_architecture = pd.read_csv(architecture_path)
df_architecture = df_architecture[df_architecture['FREQ_DOMAIN'] != 'FITS_Legendre']#remove the rows with FREQ_DOMAIN == FITS_Legendre from df_architecture (as they are already included in df_OrEdge)
#-----------------------------------------------------#

#--------df OrEdge --------#
#--------df OrEdge --------#
# Select and filter the standard OrEdge baseline rows from df_OrEdge
df_OrEdge = pd.read_csv(OrEdge_path)
df_OrEdge_Fits_Legendre = df_OrEdge[
    (df_OrEdge['FREQ_DOMAIN'] == 'FITS_Legendre') & 
    (df_OrEdge['rec_lambda'] == 1.0) & 
    (df_OrEdge['filter_used'] == "LPF") & 
    (df_OrEdge['modules_attn'] == "linear_attn") & 
    (df_OrEdge['use_normlin'] == True)
].copy()
#-----------------------------------------------------#

#------------------------------------------------------#
# Append the exact filtered OrEdge rows to ensure identical seed matching
df_ablation = pd.concat([df_ablation, df_OrEdge_Fits_Legendre], ignore_index=True).reset_index(drop=True)
df_basis = pd.concat([df_basis, df_OrEdge_Fits_Legendre], ignore_index=True).reset_index(drop=True)
df_architecture = pd.concat([df_architecture, df_OrEdge_Fits_Legendre], ignore_index=True).reset_index(drop=True)


df_architecture['FREQ_DOMAIN'] = df_architecture['FREQ_DOMAIN'].replace({'FITS_Legendre': 'OrEdge'})

def parse_metrics(metrics_string):
    if pd.isna(metrics_string):
        return {}
    pattern = r'(\bpr|rc|auc|ap|f1):\s*([0-9.]+)'
    matches = re.findall(pattern, metrics_string)
    return {k: float(v) for k, v in matches}

# 2. Process Ablation DataFrame
#df_ablation['device'] = np.where(df_ablation['training_time_per_epoch'] == 0, 'raspberry_pi', 'server')
# 2. Process Ablation DataFrame
sub_ablation = df_ablation.copy()
# Adjust MSDS auxi_lambda parameter BEFORE dictionary normalization & key mapping
sub_ablation.loc[sub_ablation['datasource'] == 'MSDS', 'auxi_lambda'] = 0.1111

metrics_ablation = pd.json_normalize(sub_ablation['info_dict'].apply(parse_metrics))
sub_ablation = pd.concat([sub_ablation.reset_index(drop=True), metrics_ablation.reset_index(drop=True)], axis=1)
# Dataset-specific FreDF weight
dataset_auxi_lambda = {
    'SN': 0.5,
    'TT': 0.2,
    'MSDS': 0.1111,
}

normlin_col = 'use_normlin' if 'use_normlin' in sub_ablation.columns else 'NormLin'

# Build the mapping
config_to_name = {}

# without FreDF loss -> always auxi_lambda = 0.0
config_to_name[(1.0, 0.0, "LPF", "linear_attn", True)] = "without FreDF loss"

# Remaining variants -> dataset-dependent auxi_lambda
for aux in dataset_auxi_lambda.values():
    config_to_name[(1.0, aux, "LPF", "no_attn", True)] = "without linear-attn"
    config_to_name[(1.0, aux, "LPF", "linear_attn", False)] = "without NormLin"
    config_to_name[(1.0, aux, "nofilter", "linear_attn", True)] = "without low-pass filter"
    config_to_name[(1.0, aux, "LPF", "linear_attn", True)] = "OrEdge (linear attn, FreDF)"

# Dataset-dependent auxi_lambda for matching
sub_ablation['auxi_lambda_key'] = sub_ablation['datasource'].map(dataset_auxi_lambda)

# Override for rows where FreDF is disabled (auxi_lambda should always be 0)
sub_ablation.loc[sub_ablation['auxi_lambda'] == 0, 'auxi_lambda_key'] = 0.0

sub_ablation['ablation_key'] = list(zip(
    sub_ablation['rec_lambda'],
    sub_ablation['auxi_lambda_key'],
    sub_ablation['filter_used'],
    sub_ablation['modules_attn'],
    sub_ablation[normlin_col]
))

sub_ablation['Row_Label'] = sub_ablation['ablation_key'].map(config_to_name)
sub_ablation = sub_ablation.dropna(subset=['Row_Label'])


# 3. Process Basis DataFrame
#df_basis['device'] = np.where(df_basis['training_time_per_epoch'] == 0, 'raspberry_pi', 'server')
sub_basis = df_basis   #[(df_basis['device'] == 'server') & (df_basis['FREQ_DOMAIN'] == 'FITS_Legendre')].copy()
metrics_basis = pd.json_normalize(sub_basis['info_dict'].apply(parse_metrics))
sub_basis = pd.concat([sub_basis.reset_index(drop=True), metrics_basis.reset_index(drop=True)], axis=1)

rename_basis = {
    "hermite": "Hermite Basis",
    "chebyshev": "Chebyshev Basis",
    "laguerre": "Laguerre Basis",
    "fourier": "Fourier Basis",
    "legendre": "OrEdge (with legendre)"
}
sub_basis['Row_Label'] = sub_basis['basis_type'].map(rename_basis)
sub_basis = sub_basis.dropna(subset=['Row_Label'])


# 3.5 Process Architecture DataFrame
metrics_architecture = pd.json_normalize(df_architecture['info_dict'].apply(parse_metrics))
sub_architecture = pd.concat([df_architecture.reset_index(drop=True), metrics_architecture.reset_index(drop=True)], axis=1)
sub_architecture['Row_Label'] = sub_architecture['FREQ_DOMAIN']

oredge_msds_exact = sub_architecture[
    (sub_architecture['Row_Label'] == 'OrEdge') & 
    (sub_architecture['datasource'] == 'MSDS')
].copy()
oredge_msds_exact['Row_Label'] = 'OrEdge (linear attn, FreDF)'
sub_ablation = sub_ablation[~((sub_ablation['Row_Label'] == 'OrEdge (linear attn, FreDF)') & (sub_ablation['datasource'] == 'MSDS'))]
sub_ablation = pd.concat([sub_ablation, oredge_msds_exact], ignore_index=True).reset_index(drop=True)


# 4. Matrix & Shading Parameters
datasets = ['MSDS','SN','TT']
metrics = [("PR", "pr"), ("RC", "rc"), ("F1", "f1"), ("AUC", "auc"), ("AP", "ap")]

ablation_order = [
    'without FreDF loss', 
    'without linear-attn', 
    #'without NormLin', 
    #'without low-pass filter',
    'OrEdge (linear attn, FreDF)'
]
basis_order = [
    'Hermite Basis', 
    'Chebyshev Basis',
    'Laguerre Basis', 
    'Fourier Basis', 
    'OrEdge (with legendre)'
]
architecture_order = [
    'FEDformerModel',
    'iTransformer', 
    #'DLinear', 
    #'FreTS',   
    'OrEdge'  
]

def get_accuracy_color_macro(val, min_val, max_val):
    if pd.isna(val) or min_val == max_val or np.isnan(min_val) or np.isnan(max_val):
        return ""
    norm = (val - min_val) / (max_val - min_val)
    max_opacity = 35
    if norm > 0.5:
        return f"\\cellcolor{{green!{int((norm - 0.5) * 2 * 100)}!yellow!{max_opacity}}}"
    return f"\\cellcolor{{yellow!{int(norm * 2 * 100)}!red!{max_opacity}}}"

# Modified to extract and normalize bounding parameters LOCAL to the passed dataframe
def generate_local_block_rows(order_list, source_df):
    block_lines = []
    for method in order_list:
        row_cells = []
        for ds in datasets:
            for metric_label, csv_col in metrics:
                
                # --- LOCAL BOUNDING CALCULATION ---
                # Computes min and max ONLY from the current method group to maximize cell contrast
                all_means = [
                    source_df[(source_df['Row_Label'] == m) & (source_df['datasource'] == ds)][csv_col].mean() 
                    for m in order_list
                ]
                all_means = [m for m in all_means if not pd.isna(m)]
                min_val = min(all_means) if all_means else np.nan
                max_val = max(all_means) if all_means else np.nan
                
                cell_data = source_df[(source_df['Row_Label'] == method) & (source_df['datasource'] == ds)][csv_col]
                if not cell_data.empty and not pd.isna(cell_data.mean()):
                    mean_val = cell_data.mean()
                    std_val = cell_data.std()
                    color_prefix = get_accuracy_color_macro(mean_val, min_val, max_val)
                    std_str = f"{std_val:.3f}" if (not pd.isna(std_val) and std_val > 0) else "0.000"
                    row_cells.append(f"{color_prefix}\\shortstack{{{mean_val:.3f}\\\\ \\tiny$\\pm${std_str}}}")
                else:
                    row_cells.append("--")
        cells_str = " & ".join(row_cells)
        block_lines.append(f"{method:<50} & {cells_str} \\\\")
    return block_lines

# 5. Generate Latex Frame
latex_table = []
latex_table.append("\\begin{table*}[t]\n\\centering\n\\caption{Ablation Analysis of Architectural Components and Orthogonal Transformation Bases (Mean $\\pm$ SD)}\n\\label{tab:comprehensive_rq2}\n\\scriptsize\n\\setlength{\\tabcolsep}{2pt}")

# --- MODIFIED: Automatically generates vertical lines 'l| ccccc | ccccc | ccccc' ---
column_spec = "l|" + " | ".join(["ccccc" for _ in datasets])
latex_table.append(f"\\begin{{tabular}}{{{column_spec}}}\n\\toprule")

dataset_headers = " & ".join([f"\\backslash multicolumn{{5}}{{c}}{{\\textbf{{{ds}}}}}" for ds in datasets]).replace('\\backslash ', '\\')
# add * beside MSDS in the header to indicate that it is a low-data regime dataset
dataset_headers = dataset_headers.replace('MSDS', 'MSDS*')


latex_table.append(f"\\textbf{{Configuration Group}} & {dataset_headers} \\\\")

cmidrules = [f"\\cmidrule(lr){{{i*5+2}-{i*5+6}}}" for i in range(len(datasets))]
latex_table.append(" ".join(cmidrules))

metric_labels = " & ".join([m[0] for m in metrics] * len(datasets))
latex_table.append(f" & {metric_labels} \\\\\n\\midrule")

# Append Architectural Block with localized bounds
latex_table.append("\\multicolumn{16}{l}{\\textbf{(a) Impact of Architectural Components}} \\\\")
latex_table.extend(generate_local_block_rows(ablation_order, sub_ablation))
latex_table.append("\\midrule[2pt]")

# Append Functional Basis Block with localized bounds
#latex_table.append("\\multicolumn{16}{l}{\\textbf{(b) Impact of Alternative Functional Projection Bases}} \\\\")
#latex_table.extend(generate_local_block_rows(basis_order, sub_basis))
#latex_table.append("\\midrule[2pt]")

# Append Architecture Block with localized bounds
latex_table.append("\\multicolumn{16}{l}{\\textbf{(b) Impact of Architectural Variants}} \\\\")
latex_table.extend(generate_local_block_rows(architecture_order, sub_architecture))


latex_table.append("\\bottomrule\n\\end{tabular}\n")
# add flushleft at the end of the table
flushleft_note = """
	\\begin{flushleft}
		*For MSDS, from RQ5, we find that orthogonal-domain supervision provides limited benefit, so we omit the FreDF loss in the ablation study, all experiments are conducted with $\lambda_{aux}=0$. 
	\end{flushleft}
"""
latex_table.append(flushleft_note)
latex_table.append("\\end{table*}")

with open('sections/Results/RQ3_ablations.tex', 'w') as f:
    f.write("\n".join(latex_table))
print("LaTeX table saved to sections/Results/RQ3_ablations.tex !!")



# Function to check seeds for a specific DataFrame, group column, and required order
def check_seeds(df, name_col, order_list, group_title):
    print(f"\n================ Verification for {group_title} ================")
    missing_found = False
    
    for ds in datasets:
        for item in order_list:
            subset = df[(df['datasource'] == ds) & (df[name_col] == item)]
            
            # Check if seeds column exists
            if 'random_seed' in subset.columns:
                seeds = subset['random_seed'].dropna().unique()
            elif 'seed' in subset.columns:
                seeds = subset['seed'].dropna().unique()
            else:
                seeds = []

            seed_count = len(seeds)
            if seed_count < 3:
                missing_found = True
                print(f"[MISSING] Datasource: {ds:<5} | Item: {item:<30} | Found Seeds ({seed_count}/3): {list(seeds)}")
                
    if not missing_found:
        print(f"All items in {group_title} have at least 3 seeds across all datasets!")

# Run checks for all three groups
check_seeds(sub_ablation, 'Row_Label', ablation_order, "Ablation Group")
check_seeds(sub_basis, 'Row_Label', basis_order, "Basis Group")
check_seeds(sub_architecture, 'Row_Label', architecture_order, "Architecture Group")