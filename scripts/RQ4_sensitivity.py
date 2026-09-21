import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Load data
# =============================================================================

path = "result_journal/result_RQ3_sensitivity_auxi_lambda_lowshow_simplegraph.csv"
df = pd.read_csv(path)

os.makedirs("figs/results", exist_ok=True)

# =============================================================================
# Parse evaluation metrics
# =============================================================================

def parse_metrics(metrics_string):
    if pd.isna(metrics_string):
        return {}

    pattern = r'(\bpr|rc|auc|ap|f1):\s*([0-9.]+)'
    return {k: float(v) for k, v in re.findall(pattern, metrics_string)}

metrics_df = pd.json_normalize(df["info_dict"].apply(parse_metrics))
df = pd.concat([df.reset_index(drop=True),
                metrics_df.reset_index(drop=True)], axis=1)

# =============================================================================
# Plot configuration
# =============================================================================

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

dataset_colors = {
    "SN": "#1b8583",
    "MSDS": "#e6a100",
    "TT": "#8a63d2"
}

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

    fig, ax = plt.subplots(figsize=(2.8, 2.4))

    x_ticks = None

    ymin = 1
    ymax = 0

    for metric, label, marker in metrics:

        grouped = (
            ds_df.groupby("auxi_lambda")[metric]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("auxi_lambda")
        )

        grouped["std"] = grouped["std"].fillna(0)

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

        ymin = min(ymin, (grouped["mean"]-grouped["std"]).min())
        ymax = max(ymax, (grouped["mean"]+grouped["std"]).max())

        x_ticks = grouped["auxi_lambda"]

    ax.set_xticks(np.arange(len(x_ticks)))
    ax.set_xticklabels(x_ticks.astype(str))

    ax.set_xlabel(r'Auxiliary Loss Weight ($\lambda_{auxi}$)')
    ax.set_ylabel("Score")

    padding = max((ymax-ymin)*0.2,0.02)
    ax.set_ylim(max(0,ymin-padding),min(1,ymax+padding))

    ax.grid(True)
    ax.legend(frameon=False, fontsize=8)

    plt.savefig(
        f"figs/results/sensitivity_auxi_lambda_PRF_{ds}.pdf",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # combine all metrics to the same txt file 
    """
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
    """
    data += f"Dataset: {ds}\n"
    for metric, label, _ in metrics:
        grouped = (
            ds_df.groupby("auxi_lambda")[metric]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("auxi_lambda")
        )
        data += f"{label}:\n"
        for _, row in grouped.iterrows():
            data += f"  Auxi Lambda: {row['auxi_lambda']}, Mean: {row['mean']:.4f}, Std: {row['std']:.4f}\n"
    
    print(f"Saved figs/results/sensitivity_auxi_lambda_PRF_{ds}.pdf")

txt_filename = "figs/results/sensitivity_auxi_lambda_PRF.txt"
with open(txt_filename, "w") as f:
    f.write(data)


#across datasources, FREQ_DOMAIN (check if it has 3 seeds using "random_seed" column)
# print missing combinations with the found seeds 
missing_combinations = []
for ds in df["datasource"].unique():
    for aux in df["auxi_lambda"].unique():
            seeds = df[
                (df["datasource"] == ds) &
                (df["auxi_lambda"] == aux)
            ]["random_seed"].unique()
            if len(seeds) < 3:
                missing_combinations.append((ds, aux, seeds))
for ds, aux, seeds in missing_combinations:
    print(f"Missing combination: datasource={ds}, aux={aux}, found seeds={seeds}")
if len(missing_combinations) == 0:
    print("All combinations have at least 3 seeds.")