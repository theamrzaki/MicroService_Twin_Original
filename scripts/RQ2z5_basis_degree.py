import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# =============================================================================
# Load data
# =============================================================================

path = "result_journal/result_RQ2z5_ablations_basis_degrees.csv" # for different degrees (3,7), except for MSDS dataset, and except degree 5 (as it comes from basis ablations)
path_RQ1 = "result_journal/result_RQ2_basis_withNorm_accMem_updatedFreDF_lowshow_simplegraph.csv" # for different basis degree5 (except for legendre as it comes from RQ1)
OrEdge_path = 'result_journal/result_RQ1_benchmark_MemOptimised_lowshow_simplegraph.csv' # for degree 5 legendre (as it comes from RQ1)

df = pd.read_csv(path)
df = df[~df["datasource"].isin(["MSDS"])]
path_MSDS = "result_journal/result_RQ2z5_ablations_basis_degrees_MSDS.csv"# for MSDS
df_MSDS = pd.read_csv(path_MSDS)
df = pd.concat([df, df_MSDS], ignore_index=True)

df_RQ1 = pd.read_csv(path_RQ1)
df_RQ1 = df_RQ1[~df_RQ1["basis_type"].isin(["legendre"])]
df_OrEdge = pd.read_csv(OrEdge_path)

# Append RQ1 benchmark data
df = pd.concat([df, df_RQ1, df_OrEdge], ignore_index=True)

# Keep the relevant frequency domain
df = df[df["FREQ_DOMAIN"].isin(["FITS_Legendre"])].copy()
df = df[df["degree"].isin([3,5,7])].copy()
os.makedirs("figs/results", exist_ok=True)

#across degress, datasources, basis_type (check if it has 3 seeds using "random_seed" column)
# print missing combinations with the found seeds 
missing_combinations = []
for ds in df["datasource"].unique():
    for basis in df["basis_type"].unique():
        for degree in df["degree"].unique():
            seeds = df[
                (df["datasource"] == ds) &
                (df["basis_type"] == basis) &
                (df["degree"] == degree)
            ]["random_seed"].unique()
            if len(seeds) < 3:
                missing_combinations.append((ds, basis, degree, seeds))
for ds, basis, degree, seeds in missing_combinations:
    print(f"Missing combination: datasource={ds}, basis_type={basis}, degree={degree}, found seeds={seeds}")
if len(missing_combinations) == 0:
    print("All combinations have at least 3 seeds.")


# =============================================================================
# Parse evaluation metrics
# =============================================================================

def parse_metrics(metrics_string):
    if pd.isna(metrics_string):
        return {}

    pattern = r'(\bpr|rc|auc|ap|f1):\s*([0-9.]+)'
    return {k: float(v) for k, v in re.findall(pattern, metrics_string)}


metrics_df = pd.json_normalize(
    df["info_dict"].apply(parse_metrics)
)

df = pd.concat(
    [
        df.reset_index(drop=True),
        metrics_df.reset_index(drop=True)
    ],
    axis=1
)

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

metrics = [
    ("pr", "Precision", "o"),
    ("rc", "Recall", "s"),
    ("f1", "F1-score", "^"),
    #("auc", "AUC", "D"),
    #("ap", "Average Precision", "v")
]

target_datasets = ["SN", "MSDS", "TT"]

basis_types = [
    "fourier",
    "hermite",
    "laguerre",
    "legendre",
    "chebyshev"
]

# =============================================================================
# Output PDF
# =============================================================================

pdf_filename = (
    "figs/results/"
    "sensitivity_degree_by_basis.pdf"
)

# =============================================================================
# Generate one PDF page per dataset
# =============================================================================

data = ""

with PdfPages(pdf_filename) as pdf:

    for ds in target_datasets:

        # ---------------------------------------------------------------------
        # Create one page for this dataset
        # ---------------------------------------------------------------------

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(8.5, 5.8)
        )

        axes = axes.flatten()

        # ---------------------------------------------------------------------
        # Plot each basis
        # ---------------------------------------------------------------------

        for basis_idx, basis in enumerate(basis_types):

            ax = axes[basis_idx]

            ds_basis_df = df[
                (df["datasource"] == ds) &
                (df["basis_type"].str.lower() == basis)
            ].copy()

            if ds_basis_df.empty:
                ax.set_visible(False)
                print(f"No data found for {ds} / {basis}")
                continue

            ymin = 1.0
            ymax = 0.0

            x_ticks = None

            # -------------------------------------------------------------
            # Plot all metrics
            # -------------------------------------------------------------

            for metric, label, marker in metrics:

                if metric not in ds_basis_df.columns:
                    continue

                grouped = (
                    ds_basis_df
                    .groupby("degree")[metric]
                    .agg(["mean", "std"])
                    .reset_index()
                    .sort_values("degree")
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
                    label=label
                )

                ymin = min(
                    ymin,
                    (grouped["mean"] - grouped["std"]).min()
                )

                ymax = max(
                    ymax,
                    (grouped["mean"] + grouped["std"]).max()
                )

                x_ticks = grouped["degree"].values

            # -------------------------------------------------------------
            # Axis configuration
            # -------------------------------------------------------------

            if x_ticks is not None:
                ax.set_xticks(np.arange(len(x_ticks)))
                ax.set_xticklabels(x_ticks.astype(str))
            #if basis!= "hermite":
            #    ax.set_yticks(np.arange(0, 1.1, 0.01))
            ax.set_xlabel("Projection Order")
            ax.set_ylabel("Score")

            ax.set_title(
                basis.capitalize(),
                fontsize=11
            )

            import matplotlib.ticker as ticker

            # ---------------------------------------------------------------------
            # Y-axis range & Formatting
            # ---------------------------------------------------------------------
            #padding = max((ymax - ymin) * 0.2, 0.02)

            # Ensure upper bound strictly never exceeds 1.0
            lower_limit = max(0.0, ymin)
            upper_limit = min(1.0, ymax)

            ax.set_ylim(lower_limit, upper_limit)

            # Force tick marks to stay strictly within [0, 1]
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

            # Remove any auto-generated ticks that end up above 1.0
            yticks = [t for t in ax.get_yticks() if 0.0 <= t <= 1.0]
            ax.set_yticks(yticks)

            ax.grid(True)

            ax.legend(
                frameon=False,
                fontsize=6.5,
                loc="best"
            )

            # -------------------------------------------------------------
            # Save numerical results
            # -------------------------------------------------------------

            data += f"Dataset: {ds}\n"
            data += f"Basis: {basis}\n"

            for metric, label, _ in metrics:

                if metric not in ds_basis_df.columns:
                    continue

                grouped = (
                    ds_basis_df
                    .groupby("degree")[metric]
                    .agg(["mean", "std"])
                    .reset_index()
                    .sort_values("degree")
                )

                data += f"{label}:\n"

                for _, row in grouped.iterrows():

                    data += (
                        f"  Degree: {int(row['degree'])}, "
                        f"Mean: {row['mean']:.4f}, "
                        f"Std: {row['std']:.4f}\n"
                    )

            data += "\n"

        # ---------------------------------------------------------------------
        # Hide unused sixth subplot
        # ---------------------------------------------------------------------

        axes[-1].set_visible(False)

        # ---------------------------------------------------------------------
        # Dataset title
        # ---------------------------------------------------------------------

        fig.suptitle(
            ds,
            fontsize=14,
            fontweight="bold"
        )

        plt.tight_layout(
            rect=[0, 0, 1, 0.95]
        )

        # ---------------------------------------------------------------------
        # Add page to PDF
        # ---------------------------------------------------------------------

        pdf.savefig(
            fig,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

        print(f"Added {ds} page to PDF")

# =============================================================================
# Save numerical results
# =============================================================================

txt_filename = (
    "figs/results/"
    "sensitivity_degree_by_basis.txt"
)

with open(txt_filename, "w") as f:
    f.write(data)

print(f"Saved {pdf_filename}")
print(f"Saved {txt_filename}")




# =============================================================================
# Paper-ready individual figures
# One PDF per (dataset, basis)
# =============================================================================

paper_fig_dir = "figs/results/sensitivity_individual"
os.makedirs(paper_fig_dir, exist_ok=True)

for ds in target_datasets:

    for basis in basis_types:

        # ---------------------------------------------------------------------
        # Filter dataset + basis
        # ---------------------------------------------------------------------

        ds_basis_df = df[
            (df["datasource"] == ds) &
            (df["basis_type"].str.lower() == basis)
        ].copy()

        if ds_basis_df.empty:
            print(f"No data found for {ds} / {basis}")
            continue

        # ---------------------------------------------------------------------
        # Create figure
        # ---------------------------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(6.5, 4.5)
        )

        ymin = 1.0
        ymax = 0.0
        x_ticks = None

        # ---------------------------------------------------------------------
        # Plot all metrics
        # ---------------------------------------------------------------------

        for metric, label, marker in metrics:

            if metric not in ds_basis_df.columns:
                continue

            grouped = (
                ds_basis_df
                .groupby("degree")[metric]
                .agg(["mean", "std"])
                .reset_index()
                .sort_values("degree")
            )

            grouped["std"] = grouped["std"].fillna(0)

            x = np.arange(len(grouped))

            ax.errorbar(
                x,
                grouped["mean"],
                yerr=grouped["std"],
                marker=marker,
                linewidth=5,
                capsize=5,
                markersize=7,
                label=label
            )

            ymin = min(
                ymin,
                (grouped["mean"] - grouped["std"]).min()
            )

            ymax = max(
                ymax,
                (grouped["mean"] + grouped["std"]).max()
            )

            x_ticks = grouped["degree"].values

        # ---------------------------------------------------------------------
        # Axis configuration
        # ---------------------------------------------------------------------

        if x_ticks is not None:

            ax.set_xticks(
                np.arange(len(x_ticks))
            )

            ax.set_xticklabels(
                x_ticks.astype(str),
                fontsize=17
            )

        ax.set_xlabel(
            "Projection Order",
            fontsize=18
        )

        ax.set_ylabel(
            "Score",
            fontsize=18
        )

        ax.set_yticklabels(
            ax.get_yticks().round(2).astype(str),
            fontsize=17
        )
        # ---------------------------------------------------------------------
        # Title
        # ---------------------------------------------------------------------

        ax.set_title(
            f"{ds} - {basis.capitalize()}",
            fontsize=15,
            fontweight="bold"
        )

        # ---------------------------------------------------------------------
        # Y-axis range
        # ---------------------------------------------------------------------

        import matplotlib.ticker as ticker

        # ---------------------------------------------------------------------
        # Y-axis range & Formatting
        # ---------------------------------------------------------------------
        #padding = max((ymax - ymin) * 0.2, 0.02)

        # Ensure upper bound strictly never exceeds 1.0
        lower_limit = max(0.0, ymin)
        upper_limit = min(1.0, ymax)

        ax.set_ylim(lower_limit, upper_limit)

        # Force tick marks to stay strictly within [0, 1]
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

        # Remove any auto-generated ticks that end up above 1.0
        yticks = [t for t in ax.get_yticks() if 0.0 <= t <= 1.0]
        ax.set_yticks(yticks)

        # ---------------------------------------------------------------------
        # Grid and legend
        # ---------------------------------------------------------------------

        ax.grid(True)

        ax.legend(
            frameon=False,
            fontsize=15,
            loc="best"
        )

        # ---------------------------------------------------------------------
        # Layout
        # ---------------------------------------------------------------------

        plt.tight_layout()

        # ---------------------------------------------------------------------
        # Filename
        # ---------------------------------------------------------------------

        filename = (
            f"{ds}_{basis}_projection_order_sensitivity.pdf"
        )

        filepath = os.path.join(
            paper_fig_dir,
            filename
        )

        # ---------------------------------------------------------------------
        # Save
        # ---------------------------------------------------------------------

        fig.savefig(
            filepath,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

        print(f"Saved: {filepath}")