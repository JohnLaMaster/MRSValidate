
"""
Toy-data generator for illustrative uncertainty plots inspired by:
- Meronen et al. (predictive marginal variance / predictive entropy)
- Ghoshal et al. (aleatoric / epistemic uncertainty scatter patterns)
- Lee & Kim (aleatoric / epistemic / total uncertainty across parameter ranges)

The goal is qualitative reproduction of the visible patterns, not recovery of
the original experimental data.

Output:
  - PNG, PDF, SVG figures
  - Optional CSV exports (enabled by default)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

try:
    from scipy.stats import gaussian_kde
    from scipy import linalg as scipy_linalg
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False
    scipy_linalg = None

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

SEED = 12
RNG = np.random.default_rng(SEED)

EXPORT_CSV = True
SAVE_FORMATS = ("png", "pdf", "svg")

N_MERONEN = 1000
N_GHOSHAL = 1000#700#350
N_LEE = 500

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

OUTPUT_DIR = BASE_DIR / "mrs_toy_outputs"
FIG_DIR = OUTPUT_DIR / "figures"
CSV_DIR = OUTPUT_DIR / "csv"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

GREEN = "#34A853"
BLUE = "#4C78A8"
RED = "#E45756"
HOT_PINK = "#E43D96"
GRAY = "#BEBEBE"
DARK_GRAY = "#666666"

MERONEN_BG = "#FFFFFF"#"#F3F3F3"
GHOSHAL_BG = "#EBEBEB"
LEE_BG = "#FFFFFF"#"#F7F7F7"

plt.rcParams.update({
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

def positive_normal(rng, mean, sd, n, low=0.0, high=None):
    x = rng.normal(mean, sd, n)
    x = np.clip(x, low, np.inf if high is None else high)
    return x

def sample_mixture(rng, n, components):
    weights = np.array([w for w, _ in components], dtype=float)
    weights = weights / weights.sum()
    counts = rng.multinomial(n, weights)
    out = []
    for count, (_, sampler) in zip(counts, components):
        out.append(sampler(count))
    x = np.concatenate(out)
    rng.shuffle(x)
    return x

def save_figure(fig, stem):
    for ext in SAVE_FORMATS:
        fig.savefig(FIG_DIR / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)

def export_csv(df, name):
    if EXPORT_CSV:
        df.to_csv(CSV_DIR / f"{name}.csv", index=False)
        
def _profile_from_anchors(bin_centers, anchors):
    """
    Interpolate a desired per-bin display mass from hand-specified anchors.

    anchors:
        list of (x, display_height) tuples

    The display heights are deliberately on the approximate scale visible
    in the Meronen figure (0--0.3), rather than PDF-density units.
    """
    anchors = np.asarray(anchors, dtype=float)

    return np.interp(bin_centers, anchors[:, 0], anchors[:, 1], left=0.0, right=0.0)


def _sample_from_profile(rng, n, bin_edges, profile):
    """
    Generate n raw toy observations from a desired histogram profile.

    The profile controls the relative probability of selecting each bin.
    Once a bin is selected, the observation is jittered uniformly within it.
    """
    profile = np.asarray(profile, dtype=float)
    if np.any(profile < 0):
        raise ValueError("Profile values must be non-negative.")
    probabilities = profile / profile.sum()

    selected_bins = rng.choice(len(profile), size=n, replace=True, p=probabilities)
    left = bin_edges[selected_bins]
    right = bin_edges[selected_bins + 1]
    return rng.uniform(left, right)


def _smooth_profile(x, y, sigma_points=5):
    """
    Very light smoothing for the display line.

    Unlike gaussian_kde(), this does NOT globally refit the distribution,
    so it preserves plateaus, bridge mass, and separated peaks.
    """
    y = np.asarray(y, dtype=float)
    radius = int(4 * sigma_points)
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma_points) ** 2)
    kernel /= kernel.sum()

    # Pad with edge values so the near-zero peak is not artificially
    # pulled downward by zero padding.
    padded = np.pad(y, radius, mode="edge")
    return np.convolve(padded, kernel, mode="same")[radius:-radius]


# -------------------------------------------------------------------------
# Meronen-style toy data
# -------------------------------------------------------------------------

def get_meronen_variance_anchors(panel_idx):
    """Hand-tuned visual profiles for Meronen predictive marginal variance panels 1-3."""

    if panel_idx == 1:
        return {
            "CIFAR-10": [
                (0.00000, 0.34),
                (0.00015, 0.34),
                (0.00030, 0.30),
                (0.00045, 0.16),
                (0.00065, 0.065),
                (0.00090, 0.020),
                (0.00120, 0.006),
                (0.00160, 0.001),
                (0.00200, 0.000),
                (0.02500, 0.000),
            ],

            "CIFAR-100": [
                (0.00000, 0.135),
                (0.00025, 0.110),
                (0.00055, 0.070),
                (0.00090, 0.050),
                (0.00140, 0.041),
                (0.00220, 0.036),
                (0.00320, 0.038),
                (0.00420, 0.047),
                (0.00520, 0.058),
                (0.00620, 0.066),
                (0.00700, 0.069),
                (0.00780, 0.066),
                (0.00880, 0.055),
                (0.01000, 0.038),
                (0.01150, 0.024),
                (0.01300, 0.016),
                (0.01500, 0.010),
                (0.01800, 0.006),
                (0.02100, 0.003),
                (0.02500, 0.000),
            ],

            "SVHN": [
                (0.00000, 0.300),
                (0.00020, 0.270),
                (0.00045, 0.215),
                (0.00080, 0.175),
                (0.00120, 0.145),
                (0.00180, 0.115),
                (0.00250, 0.085),
                (0.00330, 0.065),
                (0.00420, 0.050),
                (0.00520, 0.040),
                (0.00650, 0.030),
                (0.00800, 0.021),
                (0.00950, 0.013),
                (0.01100, 0.007),
                (0.01300, 0.003),
                (0.01500, 0.000),
                (0.02500, 0.000),
            ],
        }

    if panel_idx == 2:
        return {
            "CIFAR-10": [
                (0.00000, 0.34),
                (0.00012, 0.34),
                (0.00025, 0.275),
                (0.00040, 0.120),
                (0.00060, 0.040),
                (0.00085, 0.012),
                (0.00110, 0.003),
                (0.00140, 0.000),
                (0.02500, 0.000),
            ],

            "CIFAR-100": [
                (0.00000, 0.340),
                (0.00012, 0.300),
                (0.00025, 0.175),
                (0.00045, 0.085),
                (0.00070, 0.055),
                (0.00100, 0.045),
                (0.00150, 0.038),
                (0.00220, 0.033),
                (0.00300, 0.028),
                (0.00400, 0.020),
                (0.00500, 0.014),
                (0.00600, 0.008),
                (0.00700, 0.004),
                (0.00800, 0.001),
                (0.00900, 0.000),
                (0.02500, 0.000),
            ],

            "SVHN": [
                (0.00000, 0.185),
                (0.00015, 0.150),
                (0.00030, 0.080),
                (0.00050, 0.045),
                (0.00075, 0.038),
                (0.00100, 0.050),
                (0.00130, 0.070),
                (0.00160, 0.088),
                (0.00185, 0.095),
                (0.00210, 0.088),
                (0.00240, 0.070),
                (0.00280, 0.050),
                (0.00320, 0.033),
                (0.00370, 0.020),
                (0.00430, 0.010),
                (0.00500, 0.003),
                (0.00570, 0.000),
                (0.02500, 0.000),
            ],
        }

    if panel_idx == 3:
        return {
            "CIFAR-10": [
                (0.00000, 0.000),
                (0.00018, 0.210),
                (0.00032, 0.340),
                (0.00050, 0.300),
                (0.00075, 0.195),
                (0.00100, 0.095),
                (0.00130, 0.040),
                (0.00170, 0.012),
                (0.00210, 0.003),
                (0.00250, 0.000),
                (0.02500, 0.000),
            ],

            "CIFAR-100": [
                (0.00000, 0.000),
                (0.00018, 0.200),
                (0.00032, 0.340),
                (0.00050, 0.300),
                (0.00075, 0.200),
                (0.00100, 0.100),
                (0.00130, 0.050),
                (0.00180, 0.042),
                (0.00300, 0.038),
                (0.00650, 0.034),
                (0.00900, 0.030),
                (0.01100, 0.025),
                (0.01300, 0.018),
                (0.01600, 0.010),
                (0.01900, 0.004),
                (0.02200, 0.001),
                (0.02500, 0.000),
            ],

            "SVHN": [
                (0.00000, 0.000),
                (0.00018, 0.160),
                (0.00032, 0.300),
                (0.00050, 0.260),
                (0.00075, 0.150),
                (0.00100, 0.075),
                (0.00140, 0.032),
                (0.00220, 0.026),
                (0.00350, 0.025),
                (0.00450, 0.030),
                (0.00550, 0.060),
                (0.00650, 0.095),
                (0.00730, 0.090),
                (0.00820, 0.070),
                (0.00920, 0.045),
                (0.01050, 0.025),
                (0.01200, 0.010),
                (0.01400, 0.002),
                (0.01600, 0.000),
                (0.02500, 0.000),
            ],
        }

    raise ValueError("panel_idx must be 1, 2, or 3.")

def generate_meronen_variance_profile_data(panel_idx, n=N_MERONEN, rng=RNG, bins=32, xlim=(0.0, 0.025)):
    anchors = get_meronen_variance_anchors(panel_idx)
    bin_edges = np.linspace(xlim[0], xlim[1], bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    profiles = {}
    data = {}

    for label, anchor_values in anchors.items():
        profile = _profile_from_anchors(bin_centers, anchor_values)
        profiles[label] = profile
        data[label] = _sample_from_profile(rng, n, bin_edges, profile)

    return data, profiles, anchors, bin_edges

def plot_meronen_variance_profile(panel_idx, data, profiles, anchors, bin_edges):
    stem = f"meronen_{panel_idx:02d}_predictive_marginal_variance"

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.set_facecolor(MERONEN_BG)

    color_map = {"CIFAR-10": GREEN, "CIFAR-100": BLUE, "SVHN": RED}
    dense_x = np.linspace(0.0, 0.025, 1200)
    export_parts = []

    for label in ["CIFAR-10", "CIFAR-100", "SVHN"]:
        x = data[label]
        profile = profiles[label]
        color = color_map[label]

        display_total = profile.sum()
        sample_weights = np.full(len(x), display_total / len(x))

        ax.hist(x, bins=bin_edges, weights=sample_weights, alpha=0.18, color=color, edgecolor="none")

        anchor_array = np.asarray(anchors[label], dtype=float)
        dense_profile = np.interp(dense_x, anchor_array[:, 0], anchor_array[:, 1], left=0.0, right=0.0)
        # dense_profile = _smooth_profile(dense_x, dense_profile, sigma_points=5)
        smooth_sigma = 3 if panel_idx in (1, 2) else 5
        dense_profile = _smooth_profile(dense_x, dense_profile, sigma_points=smooth_sigma)

        ax.plot(dense_x, dense_profile, color=color, lw=1.4, label=label)

        export_parts.append(pd.DataFrame({
            "dataset": label,
            "value": x,
            "plot_weight": sample_weights,
            "measure": "Predictive marginal variance",
        }))

    ax.set_xlim(0.0, 0.025)
    ax.set_ylim(0.0, 0.30)
    ax.set_xticks([0.0, 0.01, 0.02])
    ax.set_xticklabels(["0", "0.01", "0.02"])
    ax.set_yticks([0.0, 0.1, 0.2, 0.3])
    ax.set_yticklabels(["0", "0.1", "0.2", "0.3"])
    ax.set_xlabel("Predictive marginal variance")
    ax.set_ylabel("")
    ax.grid(False)

    if panel_idx == 3:
        ax.legend(loc="upper right")

    fig.tight_layout()
    save_figure(fig, stem)
    export_csv(pd.concat(export_parts, ignore_index=True), stem)

def get_meronen_entropy_anchors(panel_idx):
    """Hand-tuned visual profiles for Meronen predictive entropy panels 4-6."""

    if panel_idx == 4:
        return {
            "CIFAR-10": [
                (0.00, 0.34),
                (0.03, 0.34),
                (0.05, 0.30),
                (0.08, 0.12),
                (0.12, 0.035),
                (0.18, 0.008),
                (0.25, 0.000),
                (2.30, 0.000),
            ],
            "CIFAR-100": [
                (0.00, 0.055),
                (0.06, 0.045),
                (0.12, 0.038),
                (0.22, 0.050),
                (0.30, 0.060),
                (0.38, 0.055),
                (0.50, 0.035),
                (0.70, 0.022),
                (0.95, 0.018),
                (1.15, 0.014),
                (1.40, 0.010),
                (1.70, 0.007),
                (2.00, 0.004),
                (2.30, 0.001),
            ],
            "SVHN": [
                (0.00, 0.050),
                (0.08, 0.065),
                (0.15, 0.040),
                (0.25, 0.027),
                (0.45, 0.019),
                (0.70, 0.015),
                (1.00, 0.011),
                (1.30, 0.009),
                (1.55, 0.010),
                (1.75, 0.014),
                (1.95, 0.020),
                (2.10, 0.029),
                (2.20, 0.040),
                (2.28, 0.028),
                (2.30, 0.025),
            ],
        }

    if panel_idx == 5:
        return {
            "CIFAR-10": [
                (0.00, 0.000),
                (0.05, 0.000),
                (0.07, 0.34),
                (0.10, 0.34),
                (0.13, 0.085),
                (0.18, 0.040),
                (0.24, 0.018),
                (0.35, 0.006),
                (0.55, 0.002),
                (1.20, 0.001),
                (2.30, 0.000),
            ],
            "CIFAR-100": [
                (0.00, 0.035),
                (0.06, 0.045),
                (0.13, 0.075),
                (0.20, 0.048),
                (0.30, 0.034),
                (0.50, 0.022),
                (0.70, 0.016),
                (0.90, 0.014),
                (1.05, 0.017),
                (1.25, 0.023),
                (1.45, 0.026),
                (1.60, 0.023),
                (1.75, 0.016),
                (1.90, 0.008),
                (2.05, 0.002),
                (2.18, 0.000),
                (2.30, 0.000),
            ],
            "SVHN": [
                (0.00, 0.010),
                (0.08, 0.015),
                (0.14, 0.027),
                (0.22, 0.020),
                (0.40, 0.015),
                (0.65, 0.013),
                (0.90, 0.012),
                (1.10, 0.016),
                (1.30, 0.021),
                (1.50, 0.026),
                (1.70, 0.032),
                (1.82, 0.037),
                (1.90, 0.035),
                (2.00, 0.027),
                (2.10, 0.014),
                (2.18, 0.003),
                (2.25, 0.000),
                (2.30, 0.000),
            ],
        }

    if panel_idx == 6:
        return {
            "CIFAR-10": [
                (0.00, 0.34),
                (0.02, 0.34),
                (0.05, 0.30),
                (0.08, 0.12),
                (0.12, 0.030),
                (0.18, 0.006),
                (0.25, 0.000),
                (2.30, 0.000),
            ],
            "CIFAR-100": [
                (0.00, 0.125),
                (0.04, 0.090),
                (0.08, 0.055),
                (0.15, 0.038),
                (0.30, 0.028),
                (0.50, 0.021),
                (0.70, 0.018),
                (0.85, 0.019),
                (0.98, 0.025),
                (1.10, 0.024),
                (1.25, 0.021),
                (1.40, 0.017),
                (1.55, 0.012),
                (1.70, 0.007),
                (1.90, 0.003),
                (2.10, 0.001),
                (2.20, 0.000),
                (2.30, 0.000),
            ],
            "SVHN": [
                (0.00, 0.055),
                (0.05, 0.060),
                (0.10, 0.040),
                (0.18, 0.025),
                (0.30, 0.016),
                (0.50, 0.012),
                (0.75, 0.011),
                (1.00, 0.010),
                (1.20, 0.011),
                (1.40, 0.015),
                (1.60, 0.021),
                (1.75, 0.027),
                (1.88, 0.035),
                (1.98, 0.040),
                (2.06, 0.037),
                (2.14, 0.024),
                (2.20, 0.008),
                (2.25, 0.000),
                (2.30, 0.000),
            ],
        }

    raise ValueError("panel_idx must be 4, 5, or 6.")

def generate_meronen_entropy_profile_data(panel_idx, n=N_MERONEN, rng=RNG, bins=24, xlim=(0.0, 2.3)):
    anchors = get_meronen_entropy_anchors(panel_idx)
    bin_edges = np.linspace(xlim[0], xlim[1], bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    profiles = {}
    data = {}

    for label, anchor_values in anchors.items():
        profile = _profile_from_anchors(bin_centers, anchor_values)
        profiles[label] = profile
        data[label] = _sample_from_profile(rng, n, bin_edges, profile)

    return data, profiles, anchors, bin_edges

def plot_meronen_entropy_profile(panel_idx, data, profiles, anchors, bin_edges):
    stem = f"meronen_{panel_idx:02d}_predictive_entropy"

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.set_facecolor(MERONEN_BG)

    color_map = {
        "CIFAR-10": GREEN,
        "CIFAR-100": BLUE,
        "SVHN": RED,
    }

    dense_x = np.linspace(0.0, 2.3, 1200)
    export_parts = []

    for label in ["CIFAR-10", "CIFAR-100", "SVHN"]:
        x = data[label]
        profile = profiles[label]
        color = color_map[label]

        display_total = profile.sum()
        sample_weights = np.full(len(x), display_total / len(x))

        ax.hist(
            x,
            bins=bin_edges,
            weights=sample_weights,
            alpha=0.18,
            color=color,
            edgecolor="none",
        )

        anchor_array = np.asarray(anchors[label], dtype=float)
        dense_profile = np.interp(
            dense_x,
            anchor_array[:, 0],
            anchor_array[:, 1],
            left=0.0,
            right=0.0,
        )

        dense_profile = _smooth_profile(dense_x, dense_profile, sigma_points=4)

        ax.plot(dense_x, dense_profile, color=color, lw=1.4, label=label)

        export_parts.append(pd.DataFrame({
            "dataset": label,
            "value": x,
            "plot_weight": sample_weights,
            "measure": "Predictive entropy",
        }))

    ax.set_xlim(0.0, 2.3)
    ax.set_ylim(0.0, 0.30)

    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xticklabels(["0", "0.5", "1", "1.5", "2"])

    ax.set_yticks([0.0, 0.1, 0.2, 0.3])
    ax.set_yticklabels(["0", "0.1", "0.2", "0.3"])

    ax.set_xlabel("Predictive entropy")
    ax.set_ylabel("")
    ax.grid(False)

    if panel_idx == 4:
        ax.legend(loc="upper right")

    fig.tight_layout()
    save_figure(fig, stem)
    export_csv(pd.concat(export_parts, ignore_index=True), stem)

# -------------------------------------------------------------------------
# Ghoshal-style toy data
# -------------------------------------------------------------------------

def generate_ghoshal_data(n=N_GHOSHAL, rng=RNG):
    error = sample_mixture(rng, n, [
        (0.55, lambda k: positive_normal(rng, 0.58, 0.028, k, 0.48, 0.69)),
        (0.25, lambda k: positive_normal(rng, 0.54, 0.020, k, 0.48, 0.69)),
        (0.20, lambda k: positive_normal(rng, 0.64, 0.018, k, 0.48, 0.69)),
    ])

    # Broader scatter to avoid nearly singular covariance while keeping the same pattern.
    aleatoric = 0.082 + 0.11 * (error - 0.50) + rng.normal(0, 0.0040, n)
    mn, mx = np.amin(aleatoric), np.amax(aleatoric)
    aleatoric = (aleatoric - mn) / (mx - mn) * (0.145 - 0.114) + 0.114
    # aleatoric = np.clip(aleatoric, 0.114, 0.145)

    epistemic = 0.0567 + rng.normal(0, 0.0023, n) + 0.0009 * np.sin((error - 0.48) * 30.0)
    epistemic = np.clip(epistemic, 0.050, 0.067)

    return pd.DataFrame({"error": error, "aleatoric": aleatoric, "epistemic": epistemic})

def kde2d(x, y, xmin, xmax, ymin, ymax, gridsize=120, rng=RNG):
    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, gridsize),
        np.linspace(ymin, ymax, gridsize)
    )
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])

    if SCIPY_AVAILABLE:
        try:
            kernel = gaussian_kde(values)
        except Exception:
            # add tiny jitter if the covariance matrix is close to singular
            xj = x + rng.normal(0, 1e-4, len(x))
            yj = y + rng.normal(0, 1e-4, len(y))
            values = np.vstack([xj, yj])
            kernel = gaussian_kde(values)
        zz = np.reshape(kernel(positions).T, xx.shape)
    else:
        hist, xedges, yedges = np.histogram2d(
            x, y, bins=50, range=[[xmin, xmax], [ymin, ymax]], density=True
        )
        zz = hist.T
        for _ in range(5):
            zz = (
                zz
                + np.roll(zz, 1, axis=0) + np.roll(zz, -1, axis=0)
                + np.roll(zz, 1, axis=1) + np.roll(zz, -1, axis=1)
            ) / 5.0
        xx, yy = np.meshgrid(
            0.5 * (xedges[:-1] + xedges[1:]),
            0.5 * (yedges[:-1] + yedges[1:])
        )
    return xx, yy, zz

def plot_ghoshal_left(df, stem="ghoshal_01_aleatoric_vs_error"):
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.set_facecolor(GHOSHAL_BG)
    ax.grid(True, color="white", lw=0.75)

    cmap = plt.cm.RdYlBu_r
    norm = Normalize(vmin=df["epistemic"].min(), vmax=df["epistemic"].max())
    edge_cols = cmap(norm(df["epistemic"].to_numpy()))

    ax.scatter(
        df["error"], df["aleatoric"],
        # s=10, facecolors="none", edgecolors=edge_cols, linewidths=0.6, alpha=0.65
        s=10, facecolors=edge_cols, edgecolors=edge_cols, linewidths=0.6, alpha=0.65
    )

    xx, yy, zz = kde2d(
        df["error"].to_numpy(),
        df["aleatoric"].to_numpy(),
        xmin=0.48, xmax=0.69, ymin=0.114, ymax=0.145
    )
    ax.contour(xx, yy, zz, levels=10, colors="#8A8A8A", linewidths=0.7, alpha=0.9)

    ax.set_xlabel("Prediction Error")
    ax.set_ylabel("Aleatoric Uncertainty")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("Epistemic Uncertainty")
    cbar.set_ticks([])
    cbar.outline.set_visible(False)

    fig.tight_layout()
    save_figure(fig, stem)

def plot_ghoshal_right(df, stem="ghoshal_02_epistemic_vs_error"):
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.set_facecolor(GHOSHAL_BG)
    ax.grid(True, color="white", lw=0.75)

    cmap = plt.cm.RdYlBu_r
    norm = Normalize(vmin=df["aleatoric"].min(), vmax=df["aleatoric"].max())
    edge_cols = cmap(norm(df["aleatoric"].to_numpy()))

    ax.scatter(
        df["error"], df["epistemic"],
        # s=10, facecolors="none", edgecolors=edge_cols, linewidths=0.6, alpha=0.65
        s=10, facecolors=edge_cols, edgecolors=edge_cols, linewidths=0.6, alpha=0.65
    )

    xx, yy, zz = kde2d(
        df["error"].to_numpy(),
        df["epistemic"].to_numpy(),
        xmin=0.48, xmax=0.69, ymin=0.050, ymax=0.067
    )
    ax.contour(xx, yy, zz, levels=10, colors="#8A8A8A", linewidths=0.7, alpha=0.9)

    ax.set_xlabel("Prediction Error")
    ax.set_ylabel("Epistemic Uncertainty")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("Aleatoric Uncertainty")
    cbar.set_ticks([])
    cbar.outline.set_visible(False)

    fig.tight_layout()
    save_figure(fig, stem)

# -------------------------------------------------------------------------
# Lee-style toy data
# -------------------------------------------------------------------------

def _interp_anchor_series(x, anchors):
    anchors = np.asarray(anchors, dtype=float)
    return np.interp(x, anchors[:, 0], anchors[:, 1], left=anchors[0, 1], right=anchors[-1, 1])


def _jagged_curve(base, rng, frac_noise=0.06, abs_noise=0.0025, smooth=3):
    noise = rng.normal(0.0, abs_noise + frac_noise * base, len(base))
    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        noise = np.convolve(noise, kernel, mode="same")
    return np.clip(base + noise, 0.0, None)

def get_lee_component_anchors(panel_idx):
    if panel_idx == 1:
        return {
            "xlim": (5, 40),
            "xlabel": "SNR (a.u)",
            "ylim": (0.0, 0.30),
            "invert_x": True,
            "dotted_lines": [20, 7],
            "aleatoric": [
                (5, 0.185),
                (7, 0.145),
                (10, 0.118),
                (15, 0.090),
                (20, 0.075),
                (30, 0.070),
                (40, 0.070),
            ],
            "epistemic": [
                (5, 0.060),
                (7, 0.050),
                (10, 0.052),
                (15, 0.062),
                (20, 0.070),
                (30, 0.070),
                (40, 0.070),
            ],
        }

    if panel_idx == 2:
        return {
            "xlim": (5, 40),
            "xlabel": "Linewidth (Hz)",
            "ylim": (0.0, 0.20),
            "invert_x": False,
            "dotted_lines": [9, 20],
            "aleatoric": [
                (5, 0.072),
                (8, 0.068),
                (12, 0.058),
                (16, 0.075),
                (20, 0.104),
                (30, 0.102),
                (40, 0.102),
            ],
            "epistemic": [
                (5, 0.070),
                (8, 0.062),
                (12, 0.052),
                (16, 0.050),
                (20, 0.050),
                (30, 0.058),
                (40, 0.066),
            ],
        }

    if panel_idx == 3:
        return {
            "xlim": (5, 33),
            "xlabel": "Concentration (mmol/L)",
            "ylim": (0.0, 0.30),
            "invert_x": False,
            "dotted_lines": [7, 18],
            "aleatoric": [
                (5, 0.082),
                (8, 0.088),
                (12, 0.075),
                (18, 0.060),
                (24, 0.050),
                (28, 0.040),
                (33, 0.025),
            ],
            "epistemic": [
                (5, 0.066),
                (8, 0.060),
                (12, 0.058),
                (18, 0.055),
                (24, 0.060),
                (28, 0.085),
                (33, 0.180),
            ],
        }

    raise ValueError("panel_idx must be 1, 2, or 3.")

def generate_lee_profile_df(panel_idx, n=N_LEE, rng=RNG):
    spec = get_lee_component_anchors(panel_idx)
    x = np.linspace(spec["xlim"][0], spec["xlim"][1], n)

    ale_base = _interp_anchor_series(x, spec["aleatoric"])
    epi_base = _interp_anchor_series(x, spec["epistemic"])

    if panel_idx == 1:
        ale = _jagged_curve(ale_base, rng, frac_noise=0.08, abs_noise=0.0030, smooth=2)
        epi = _jagged_curve(epi_base, rng, frac_noise=0.08, abs_noise=0.0030, smooth=2)
    elif panel_idx == 2:
        ale = _jagged_curve(ale_base, rng, frac_noise=0.06, abs_noise=0.0025, smooth=2)
        epi = _jagged_curve(epi_base, rng, frac_noise=0.06, abs_noise=0.0025, smooth=2)
    else:
        ale = _jagged_curve(ale_base, rng, frac_noise=0.07, abs_noise=0.0030, smooth=2)
        epi = _jagged_curve(epi_base, rng, frac_noise=0.08, abs_noise=0.0035, smooth=2)

    total = np.clip(ale + epi, 0.0, None)

    return pd.DataFrame({"x": x, "aleatoric": ale, "epistemic": epi, "total": total}), spec

def plot_lee_profile_panel(panel_idx, df, spec):
    stem_map = {1: "lee_01_snr", 2: "lee_02_linewidth", 3: "lee_03_concentration"}
    stem = stem_map[panel_idx]

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.set_facecolor(LEE_BG)

    ax.fill_between(df["x"], 0, df["total"], color=GRAY, alpha=0.90, linewidth=0.0)
    ax.fill_between(df["x"], 0, df["aleatoric"], color=GREEN, alpha=0.78, linewidth=0.0)
    ax.fill_between(df["x"], 0, df["epistemic"], color=HOT_PINK, alpha=0.66, linewidth=0.0)

    ax.plot(df["x"], df["aleatoric"], color=GREEN, lw=0.9)
    ax.plot(df["x"], df["epistemic"], color=HOT_PINK, lw=0.9)
    ax.plot(df["x"], df["total"], color=DARK_GRAY, lw=0.8, alpha=0.75)

    for xpos in spec["dotted_lines"]:
        ax.axvline(xpos, color="black", linestyle=":", lw=1.2)

    ax.set_xlabel(spec["xlabel"])
    ax.set_xlim(*spec["xlim"])
    ax.set_ylabel("Uncertainty / Predicted")
    ax.set_ylim(*spec["ylim"])

    if spec["invert_x"]:
        ax.invert_xaxis()

    fig.tight_layout()
    save_figure(fig, stem)
    export_csv(df, stem)

def main():
    for panel_idx in (1, 2, 3):
        data, profiles, anchors, bin_edges = generate_meronen_variance_profile_data(
            panel_idx, n=N_MERONEN, rng=RNG, bins=32
        )
        plot_meronen_variance_profile(panel_idx, data, profiles, anchors, bin_edges)

    for panel_idx in (4, 5, 6):
        data, profiles, anchors, bin_edges = generate_meronen_entropy_profile_data(
            panel_idx, n=N_MERONEN, rng=RNG, bins=24
        )
        plot_meronen_entropy_profile(panel_idx, data, profiles, anchors, bin_edges)

    ghoshal_df = generate_ghoshal_data()
    export_csv(ghoshal_df, "ghoshal_toy_data")
    plot_ghoshal_left(ghoshal_df)
    plot_ghoshal_right(ghoshal_df)

    for panel_idx in (1, 2, 3):
        df, spec = generate_lee_profile_df(panel_idx, n=N_LEE, rng=RNG)
        plot_lee_profile_panel(panel_idx, df, spec)

    print(f"Done. Figures saved to: {FIG_DIR}")
    print(f"CSV files saved to: {CSV_DIR}")
    print(f"N_MERONEN = {N_MERONEN}, N_GHOSHAL = {N_GHOSHAL}, N_LEE = {N_LEE}")

if __name__ == "__main__":
    main()
# 1417 lines --> 962 lines