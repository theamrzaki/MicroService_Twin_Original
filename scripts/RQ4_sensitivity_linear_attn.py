import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import os

path = 'result_journal/result_RQ3_sensitivity_MemOptimised_lowshow_simplegraph.csv' # Using forward slashes for cross-OS safety
df = pd.read_csv(path)

# Ensure the output directory exists
os.makedirs('figs/results', exist_ok=True)

# 2. Extract metrics from 'info_dict'
def parse_metrics(metrics_string):
    if pd.isna(metrics_string):
        return {}
    pattern = r'(\bpr|rc|auc|ap|f1):\s*([0-9.]+)'
    matches = re.findall(pattern, metrics_string)
    return {k: float(v) for k, v in matches}

parsed_metrics = df['info_dict'].apply(parse_metrics)
metrics_df = pd.json_normalize(parsed_metrics)
df = pd.concat([df.reset_index(drop=True), metrics_df.reset_index(drop=True)], axis=1)

# Configure Professional Academic Plot Settings
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "savefig.bbox": "tight"
})

# Define the color palette for the datasets matching your style reference
dataset_colors = {
    "TT": "#8a63d2",    # Elegant Purple
    "SN": "#1b8583",    # Elegant Teal
    "MSDS": "#e6a100"  # Warm Amber
}

# Generate 3 plots by looping through the targets
target_datasets = ["SN", "MSDS", "TT"]
metrics = [
    ("pr", "Precision", "o"),
    ("rc", "Recall", "s"),
    ("f1", "F1-score", "^"),
    ("auc", "AUC", "D"),
    ("ap", "Average Precision", "v")
]

target_datasets = ["SN", "MSDS", "TT"]

data = ""
for ds in target_datasets:

    ds_df = df[df["datasource"] == ds].copy()

    if ds_df.empty:
        print(f"Warning: No data found for dataset '{ds}'.")
        continue

    fig, ax = plt.subplots(figsize=(2.8, 2.4))

    ymin = 1.0
    ymax = 0.0

    for metric, label, marker in metrics:

        grouped = (
            ds_df.groupby("linear_attn_dim")[metric]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("linear_attn_dim")
        )

        grouped["std"] = grouped["std"].fillna(0.0)

        x = np.arange(len(grouped))

        ax.errorbar(
            x,
            grouped["mean"],
            yerr=grouped["std"],
            marker=marker,
            linewidth=1.5,
            capsize=3,
            label=label,
        )

        ymin = min(ymin, (grouped["mean"] - grouped["std"]).min())
        ymax = max(ymax, (grouped["mean"] + grouped["std"]).max())

    ax.set_xticks(np.arange(len(grouped)))
    ax.set_xticklabels(grouped["linear_attn_dim"].astype(str))

    ax.set_xlim(-0.4, len(grouped) - 0.6)

    ax.set_xlabel("Linear Attention Dimension")
    ax.set_ylabel("Score")

    padding = max((ymax - ymin) * 0.2, 0.02)
    ax.set_ylim(max(0.0, ymin - padding), min(1.0, ymax + padding))

    ax.grid(True)
    ax.legend(frameon=False, fontsize=8)

    filename = f"figs/results/sensitivity_linear_attn_dim_PRF_{ds}.pdf"

    plt.savefig(filename, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()

    # combine all metrics to the same txt file 
    data += f"Dataset: {ds}\n"
    for metric, label, _ in metrics:
        grouped = (
            ds_df.groupby("linear_attn_dim")[metric]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("linear_attn_dim")
        )
        data += f"{label}:\n"
        for _, row in grouped.iterrows():
            data += f"  Linear Attn Dim: {row['linear_attn_dim']}, Mean: {row['mean']:.4f}, Std: {row['std']:.4f}\n"
    print(f"Saved {filename}")

# Save the combined metrics to a single text file
txt_filename = "figs/results/sensitivity_linear_attn_dim_PRF.txt"
with open(txt_filename, "w") as f:
    f.write(data)


#across datasources, FREQ_DOMAIN (check if it has 3 seeds using "random_seed" column)
# print missing combinations with the found seeds 
missing_combinations = []
for ds in df["datasource"].unique():
    for dim in df["linear_attn_dim"].unique():
            seeds = df[
                (df["datasource"] == ds) &
                (df["linear_attn_dim"] == dim)
            ]["random_seed"].unique()
            if len(seeds) < 3:
                missing_combinations.append((ds, dim, seeds))
for ds, dim, seeds in missing_combinations:
    print(f"Missing combination: datasource={ds}, dim={dim}, found seeds={seeds}")
if len(missing_combinations) == 0:
    print("All combinations have at least 3 seeds.")