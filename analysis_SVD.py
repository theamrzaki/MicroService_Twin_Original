import os
import numpy as np
import torch
import matplotlib.pyplot as plt

import util.data_MSDS as data_loads
import util.data_Eadro as data_Eadro

# ============================================================
# Load your datasets here
# ============================================================
dataset_name = "TT"
if dataset_name == "MSDS":
    from util.parser_MSDS_MSDS import *
    processed = data_loads.Process(**args)
elif dataset_name == "SN" or dataset_name == "TT":
    processed_train,  processed_test = data_Eadro.run(dataset_name)

if dataset_name not in ["SN", "TT"]:
    full_train_data = processed.dataset[:int(len(processed.dataset) * 0.7)]
else:
    full_train_data = processed_train


# ============================================================
# Configuration
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ENERGY_LEVELS = [0.90, 0.95, 0.99]

MODALITIES = [
    #"data_node",
    "data_log",
    #"data_edge",
]

# Replace these with your actual dataset objects/names
DATASETS = {
    dataset_name: full_train_data,
}

OUTPUT_DIR = "./svd_energy_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def to_numpy(x):
    """Convert torch/numpy input to numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def prepare_matrix(x):
    x = to_numpy(x)
    x = np.squeeze(x)

    if x.ndim == 4:
        # [T, N, N, F]
        # Match OrEdge preprocessing:
        # [T, N, N, F] -> [T, N, F]
        x = x.mean(axis=2)

    if x.ndim == 3:
        # [T, N, F] -> [N, T, F] -> [N, T*F]
        T, N, F = x.shape
        x = x.transpose(1, 0, 2).reshape(N, T * F)

    else:
        raise ValueError(f"Unexpected shape: {x.shape}")

    return np.array(x, dtype=np.float32, copy=True)

def compute_svd_energy__(X):
    """
    Compute cumulative normalized spectral energy.

    X: [N, D]

    Returns:
        energy: [rank]
        singular_values: [rank]
    """
    X = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)

    # Remove NaN/Inf samples.
    if not torch.isfinite(X).all():
        return None, None

    # Centering is intentionally NOT performed.
    # This follows the usual reconstruction-energy formulation:
    # ||X - X_d||_F^2 = sum_{k>d} sigma_k^2
    s = torch.linalg.svdvals(X)

    energy_power = s.square()
    total_energy = energy_power.sum()

    if total_energy <= 1e-12:
        return None, None

    cumulative = torch.cumsum(energy_power, dim=0)
    energy = cumulative / total_energy

    return (
        energy.detach().cpu().numpy(),
        s.detach().cpu().numpy(),
    )

def compute_svd_energy(X):
    """
    Compute cumulative normalized spectral energy.

    X: [N, D]

    The columns are centered before SVD so that the analysis
    measures variation across samples rather than the global mean.

    Returns:
        energy: [rank]
        singular_values: [rank]
    """
    X = np.array(X, dtype=np.float32, copy=True)
    X = torch.from_numpy(X).to(DEVICE)

    # Remove NaN/Inf samples.
    if not torch.isfinite(X).all():
        return None, None

    # Center each feature dimension across samples.
    X = X - X.mean(dim=0, keepdim=True)

    # SVD
    s = torch.linalg.svdvals(X)

    # Spectral energy
    energy_power = s.square()
    total_energy = energy_power.sum()

    if total_energy <= 1e-12:
        return None, None

    cumulative = torch.cumsum(energy_power, dim=0)
    energy = cumulative / total_energy

    return (
        energy.detach().cpu().numpy(),
        s.detach().cpu().numpy(),
    )

def dimensions_for_energy(energy, levels=ENERGY_LEVELS):
    """Return minimum dimensions required to reach each energy level."""
    result = {}

    for level in levels:
        indices = np.where(energy >= level)[0]

        if len(indices) == 0:
            result[level] = np.nan
        else:
            result[level] = int(indices[0] + 1)

    return result


# ============================================================
# Analyze one modality of one dataset
# ============================================================

def analyze_modality(dataset, modality):
    """
    Perform per-sample SVD analysis.

    Returns a dictionary containing:
        - individual energy curves
        - mean/std energy
        - d90/d95/d99 per sample
        - dimensions summary
    """

    energy_curves = []
    singular_values_all = []

    dimensions = {
        level: []
        for level in ENERGY_LEVELS
    }

    skipped = 0

    for idx in range(len(dataset)):

        sample = dataset[idx]

        if modality not in sample:
            skipped += 1
            continue

        X = prepare_matrix(sample[modality])

        if X.shape[0] < 2 or X.shape[1] < 1:
            skipped += 1
            continue

        energy, singular_values = compute_svd_energy(X)

        if energy is None:
            skipped += 1
            continue

        energy_curves.append(energy)
        singular_values_all.append(singular_values)

        dims = dimensions_for_energy(energy)

        for level in ENERGY_LEVELS:
            dimensions[level].append(dims[level])

    if len(energy_curves) == 0:
        #raise RuntimeError(
        #    f"No valid samples found for modality '{modality}'."
        #)
        return {
            "energy_curves": [],
            "energy_matrix": np.array([]),
            "mean_energy": np.array([]),
            "std_energy": np.array([]),
            "dimensions": {level: [] for level in ENERGY_LEVELS},
            "dimension_stats": {level: {} for level in ENERGY_LEVELS},
            "num_samples": 0,
            "num_skipped": skipped,
        }

    # SVD rank can differ between samples.
    # Pad curves with their final energy (=1).
    max_rank = max(len(e) for e in energy_curves)

    energy_matrix = np.ones(
        (len(energy_curves), max_rank),
        dtype=np.float32
    )

    for i, e in enumerate(energy_curves):
        energy_matrix[i, :len(e)] = e

    mean_energy = np.mean(energy_matrix, axis=0)
    std_energy = np.std(energy_matrix, axis=0)

    # Aggregate dimension statistics.
    dimension_stats = {}

    for level in ENERGY_LEVELS:

        values = np.asarray(dimensions[level], dtype=float)

        dimension_stats[level] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
        }

    return {
        "energy_curves": energy_curves,
        "energy_matrix": energy_matrix,
        "mean_energy": mean_energy,
        "std_energy": std_energy,
        "dimensions": dimensions,
        "dimension_stats": dimension_stats,
        "num_samples": len(energy_curves),
        "num_skipped": skipped,
    }


# ============================================================
# Plot
# ============================================================

def plot_svd_energy(
    result,
    dataset_name,
    modality,
    output_dir=OUTPUT_DIR,
):
    """
    Generate one clean SVD energy plot.
    """

    mean_energy = result["mean_energy"]
    std_energy = result["std_energy"]

    d = np.arange(1, len(mean_energy) + 1)

    plt.figure(figsize=(7, 5))

    plt.plot(
        d,
        mean_energy,
        linewidth=2,
        label="Mean SVD energy",
    )

    plt.fill_between(
        d,
        np.maximum(mean_energy - std_energy, 0),
        np.minimum(mean_energy + std_energy, 1),
        alpha=0.20,
        label="±1 std",
    )

    # Energy reference lines
    for level in ENERGY_LEVELS:
        plt.axhline(
            level,
            linestyle="--",
            linewidth=1,
            alpha=0.7,
        )

        stats = result["dimension_stats"][level]

        plt.text(
            len(mean_energy) * 0.98,
            level + 0.01,
            f"{int(level * 100)}%",
            ha="right",
            va="bottom",
            fontsize=9,
        )

    plt.xlabel("Number of dimensions")
    plt.ylabel("Cumulative explained energy")

    plt.title(
        f"{dataset_name}: {modality.replace('data_', '').title()}"
    )

    plt.ylim(0, 1.02)
    plt.xlim(1, len(mean_energy))

    plt.grid(alpha=0.25)
    plt.legend(frameon=False)

    plt.tight_layout()

    filename = (
        f"{dataset_name.replace(' ', '_')}_"
        f"{modality.replace('data_', '')}_SVD_energy.pdf"
    )

    path = os.path.join(output_dir, filename)

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    return path


# ============================================================
# Run all 3 datasets × 3 modalities
# ============================================================

all_results = {}

for dataset_name, dataset in DATASETS.items():

    print("\n" + "=" * 70)
    print(dataset_name)
    print("=" * 70)

    all_results[dataset_name] = {}

    for modality in MODALITIES:

        print(f"\nAnalyzing {modality} ...")

        result = analyze_modality(
            dataset,
            modality,
        )

        all_results[dataset_name][modality] = result

        print(
            f"Valid samples: {result['num_samples']:,}"
        )

        print(
            f"Skipped samples: {result['num_skipped']:,}"
        )

        print("\nRequired dimensions:")

        for level in ENERGY_LEVELS:

            stats = result["dimension_stats"][level]

            print(
                f"  {int(level * 100)}% energy: "
                f"mean={stats['mean']:.2f}, "
                f"median={stats['median']:.2f}, "
                f"P90={stats['p90']:.2f}, "
                f"P95={stats['p95']:.2f}, "
                f"P99={stats['p99']:.2f}"
            )

        plot_svd_energy(
            result,
            dataset_name,
            modality,
        )


print("\nDone.")
print(f"Results saved to: {OUTPUT_DIR}")