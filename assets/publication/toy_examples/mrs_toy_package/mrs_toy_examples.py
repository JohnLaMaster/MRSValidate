
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
N_GHOSHAL = 350
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

MERONEN_BG = "#F3F3F3"
GHOSHAL_BG = "#EBEBEB"
LEE_BG = "#F7F7F7"

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

def gamma_sample(rng, shape, scale, n, high=None):
    x = rng.gamma(shape, scale, n)
    if high is not None:
        x = np.clip(x, 0, high)
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

def kde_line(x, xmin=None, xmax=None, points=400):
    x = np.asarray(x)
    if xmin is None:
        xmin = float(np.min(x))
    if xmax is None:
        xmax = float(np.max(x))
    grid = np.linspace(xmin, xmax, points)

    if SCIPY_AVAILABLE and np.std(x) > 0:
        kde = gaussian_kde(x)
        y = kde(grid)
    else:
        hist, edges = np.histogram(x, bins=60, range=(xmin, xmax), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        kx = np.linspace(-3, 3, 21)
        kernel = np.exp(-0.5 * kx**2)
        kernel /= kernel.sum()
        smooth = np.convolve(hist, kernel, mode="same")
        y = np.interp(grid, centers, smooth)

    return grid, y

def save_figure(fig, stem):
    for ext in SAVE_FORMATS:
        fig.savefig(FIG_DIR / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)

def export_csv(df, name):
    if EXPORT_CSV:
        df.to_csv(CSV_DIR / f"{name}.csv", index=False)

# -------------------------------------------------------------------------
# Meronen-style toy data
# -------------------------------------------------------------------------

def generate_meronen_entropy_pair(pair_idx, n=N_MERONEN, rng=RNG):
    if pair_idx == 1:
        green = gamma_sample(rng, shape=0.42, scale=0.045, n=n, high=0.20)
        blue = sample_mixture(rng, n, [
            (0.78, lambda k: gamma_sample(rng, 0.95, 0.30, k, high=2.1)),
            (0.22, lambda k: positive_normal(rng, 1.10, 0.35, k, 0.0, 2.1))
        ])
        red = sample_mixture(rng, n, [
            (0.35, lambda k: gamma_sample(rng, 0.80, 0.09, k, high=2.1)),
            (0.65, lambda k: positive_normal(rng, 1.85, 0.22, k, 0.0, 2.1))
        ])
    elif pair_idx == 2:
        green = gamma_sample(rng, shape=0.45, scale=0.040, n=n, high=0.22)
        blue = sample_mixture(rng, n, [
            (0.72, lambda k: gamma_sample(rng, 1.10, 0.27, k, high=2.2)),
            (0.28, lambda k: positive_normal(rng, 1.35, 0.30, k, 0.0, 2.2))
        ])
        red = sample_mixture(rng, n, [
            (0.18, lambda k: gamma_sample(rng, 1.00, 0.08, k, high=2.2)),
            (0.82, lambda k: positive_normal(rng, 1.75, 0.24, k, 0.0, 2.2))
        ])
    elif pair_idx == 3:
        green = gamma_sample(rng, shape=0.35, scale=0.040, n=n, high=0.18)
        blue = sample_mixture(rng, n, [
            (0.72, lambda k: gamma_sample(rng, 2.00, 0.18, k, high=2.3)),
            (0.28, lambda k: positive_normal(rng, 0.75, 0.20, k, 0.0, 2.3))
        ])
        red = sample_mixture(rng, n, [
            (0.22, lambda k: gamma_sample(rng, 0.95, 0.09, k, high=2.3)),
            (0.78, lambda k: positive_normal(rng, 2.18, 0.16, k, 0.0, 2.3))
        ])
    else:
        raise ValueError("pair_idx must be 1, 2, or 3.")
    return {"CIFAR-10": green, "CIFAR-100": blue, "SVHN": red}

def generate_meronen_variance_pair(pair_idx, n=N_MERONEN, rng=RNG):
    if pair_idx == 1:
        green = gamma_sample(rng, shape=0.33, scale=0.00035, n=n, high=0.0018)
        blue = sample_mixture(rng, n, [
            (0.70, lambda k: gamma_sample(rng, 1.10, 0.0034, k, high=0.025)),
            (0.30, lambda k: positive_normal(rng, 0.0080, 0.0025, k, 0.0, 0.025))
        ])
        red = sample_mixture(rng, n, [
            (0.25, lambda k: gamma_sample(rng, 0.75, 0.0009, k, high=0.025)),
            (0.75, lambda k: positive_normal(rng, 0.0068, 0.0013, k, 0.0, 0.025))
        ])
    elif pair_idx == 2:
        green = gamma_sample(rng, shape=0.34, scale=0.00030, n=n, high=0.0016)
        blue = sample_mixture(rng, n, [
            (0.75, lambda k: gamma_sample(rng, 0.95, 0.0024, k, high=0.023)),
            (0.25, lambda k: positive_normal(rng, 0.0045, 0.0016, k, 0.0, 0.023))
        ])
        red = positive_normal(rng, 0.0020, 0.00085, n, 0.0, 0.010)
    elif pair_idx == 3:
        green = gamma_sample(rng, shape=0.28, scale=0.00030, n=n, high=0.0015)
        blue = sample_mixture(rng, n, [
            (0.45, lambda k: gamma_sample(rng, 1.20, 0.0022, k, high=0.025)),
            (0.55, lambda k: positive_normal(rng, 0.0064, 0.0018, k, 0.0, 0.025))
        ])
        red = sample_mixture(rng, n, [
            (0.85, lambda k: gamma_sample(rng, 1.00, 0.0028, k, high=0.025)),
            (0.15, lambda k: positive_normal(rng, 0.0060, 0.0015, k, 0.0, 0.025))
        ])
    else:
        raise ValueError("pair_idx must be 1, 2, or 3.")
    return {"CIFAR-10": green, "CIFAR-100": blue, "SVHN": red}

def plot_meronen_panel(data_dict, x_label, stem, xlim, bins=25):
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.set_facecolor(MERONEN_BG)

    color_map = {"CIFAR-10": GREEN, "CIFAR-100": BLUE, "SVHN": RED}
    export_parts = []

    for label in ["CIFAR-10", "CIFAR-100", "SVHN"]:
        x = data_dict[label]
        c = color_map[label]
        ax.hist(x, bins=bins, range=xlim, density=True, alpha=0.18, color=c, edgecolor="none")
        grid, density = kde_line(x, xmin=xlim[0], xmax=xlim[1], points=500)
        ax.plot(grid, density, color=c, lw=1.25, label=label)
        export_parts.append(pd.DataFrame({"dataset": label, "value": x, "measure": x_label}))

    ax.set_xlim(*xlim)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(x_label)
    ax.set_ylabel("")
    ax.legend(loc="upper right")
    ax.grid(False)

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
    aleatoric = np.clip(aleatoric, 0.114, 0.145)

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
        s=10, facecolors="none", edgecolors=edge_cols, linewidths=0.6, alpha=0.65
    )

    xx, yy, zz = kde2d(
        df["error"].to_numpy(),
        df["aleatoric"].to_numpy(),
        xmin=0.48, xmax=0.69, ymin=0.114, ymax=0.145
    )
    ax.contour(xx, yy, zz, levels=10, colors="#8A8A8A", linewidths=0.7, alpha=0.9)

    ax.set_xlabel("error")
    ax.set_ylabel("aleatoric")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("uncertainty")
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
        s=10, facecolors="none", edgecolors=edge_cols, linewidths=0.6, alpha=0.65
    )

    xx, yy, zz = kde2d(
        df["error"].to_numpy(),
        df["epistemic"].to_numpy(),
        xmin=0.48, xmax=0.69, ymin=0.050, ymax=0.067
    )
    ax.contour(xx, yy, zz, levels=10, colors="#8A8A8A", linewidths=0.7, alpha=0.9)

    ax.set_xlabel("error")
    ax.set_ylabel("epistemic")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("uncertainty")
    cbar.outline.set_visible(False)

    fig.tight_layout()
    save_figure(fig, stem)

# -------------------------------------------------------------------------
# Lee-style toy data
# -------------------------------------------------------------------------

def _jaggedify(y, rng, strength=0.006, smooth=5):
    noise = rng.normal(0, strength, len(y))
    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        noise = np.convolve(noise, kernel, mode="same")
    out = y + noise
    return np.clip(out, 0, None)

def generate_lee_snr(n=N_LEE, rng=RNG):
    x = np.linspace(5, 40, n)
    ale_mean = (
        0.060
        + 0.006 * np.exp(-((x - 9) / 3.0)**2)
        + 0.018 * np.clip((12 - x) / 10.0, 0, None)
    )
    epi_mean = (
        0.058
        + 0.004 * np.exp(-((x - 18) / 8.0)**2)
        + 0.10 * np.clip((8 - x) / 4.0, 0, None) ** 1.8
    )
    ale = _jaggedify(ale_mean, rng, strength=0.0045, smooth=3)
    epi = _jaggedify(epi_mean, rng, strength=0.0055, smooth=3)
    total = np.clip(ale + epi, 0, None)
    return pd.DataFrame({"x": x, "aleatoric": ale, "epistemic": epi, "total": total})

def generate_lee_linewidth(n=N_LEE, rng=RNG):
    x = np.linspace(5, 40, n)
    ale_mean = (
        0.060
        + 0.040 * (1 / (1 + np.exp(-(x - 16) / 1.8)))
        + 0.008 * np.exp(-((x - 7) / 1.8)**2)
    )
    epi_mean = (
        0.070
        - 0.015 * np.exp(-((x - 13) / 4.0)**2)
        + 0.003 * (x - 20) / 20.0
    )
    ale = _jaggedify(ale_mean, rng, strength=0.0040, smooth=3)
    epi = _jaggedify(epi_mean, rng, strength=0.0055, smooth=3)
    total = np.clip(ale + epi, 0, None)
    return pd.DataFrame({"x": x, "aleatoric": ale, "epistemic": epi, "total": total})

def generate_lee_concentration(n=N_LEE, rng=RNG):
    x = np.linspace(5, 33, n)
    ale_mean = (
        0.055
        + 0.030 * np.exp(-((x - 8) / 5.0)**2)
        + 0.004 * np.cos((x - 5) / 3.0)
    )
    ale_mean -= 0.010 * np.clip((x - 18) / 10.0, 0, None)
    epi_mean = (
        0.055
        + 0.004 * np.sin(x / 3.5)
        + 0.095 * np.clip((x - 24) / 9.0, 0, None) ** 1.7
    )
    ale = _jaggedify(ale_mean, rng, strength=0.0045, smooth=3)
    epi = _jaggedify(epi_mean, rng, strength=0.0060, smooth=3)
    total = np.clip(ale + epi, 0, None)
    return pd.DataFrame({"x": x, "aleatoric": ale, "epistemic": epi, "total": total})

def plot_lee_panel(df, stem, xlabel, ylabel="Uncertainty / Predicted",
                   dotted_lines=None, ylim=(0, 0.30), invert_x=False):
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.set_facecolor(LEE_BG)

    ax.fill_between(df["x"], 0, df["total"], color=GRAY, alpha=0.90, linewidth=0.0)
    ax.fill_between(df["x"], 0, df["aleatoric"], color=GREEN, alpha=0.78, linewidth=0.0)
    ax.fill_between(df["x"], 0, df["epistemic"], color=HOT_PINK, alpha=0.66, linewidth=0.0)

    ax.plot(df["x"], df["aleatoric"], color=GREEN, lw=0.9)
    ax.plot(df["x"], df["epistemic"], color=HOT_PINK, lw=0.9)
    ax.plot(df["x"], df["total"], color=DARK_GRAY, lw=0.8, alpha=0.75)

    if dotted_lines is not None:
        for xpos in dotted_lines:
            ax.axvline(xpos, color="black", linestyle=":", lw=1.2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    if invert_x:
        ax.invert_xaxis()

    fig.tight_layout()
    save_figure(fig, stem)
    export_csv(df, stem)

def main():
    for pair_idx in (1, 2, 3):
        variance_data = generate_meronen_variance_pair(pair_idx)
        plot_meronen_panel(
            variance_data,
            x_label="Predictive marginal variance",
            stem=f"meronen_{pair_idx:02d}_predictive_marginal_variance",
            xlim=(0.0, 0.025),
            bins=26,
        )

    for pair_idx in (1, 2, 3):
        entropy_data = generate_meronen_entropy_pair(pair_idx)
        plot_meronen_panel(
            entropy_data,
            x_label="Predictive entropy",
            stem=f"meronen_{pair_idx+3:02d}_predictive_entropy",
            xlim=(0.0, 2.2),
            bins=26,
        )

    ghoshal_df = generate_ghoshal_data()
    export_csv(ghoshal_df, "ghoshal_toy_data")
    plot_ghoshal_left(ghoshal_df)
    plot_ghoshal_right(ghoshal_df)

    lee_a = generate_lee_snr()
    plot_lee_panel(
        lee_a, stem="lee_01_snr", xlabel="SNR (a.u)",
        dotted_lines=[20, 7], ylim=(0, 0.30), invert_x=True
    )

    lee_b = generate_lee_linewidth()
    plot_lee_panel(
        lee_b, stem="lee_02_linewidth", xlabel="Linewidth (Hz)",
        dotted_lines=[9, 20], ylim=(0, 0.20), invert_x=False
    )

    lee_c = generate_lee_concentration()
    plot_lee_panel(
        lee_c, stem="lee_03_concentration", xlabel="Concentration (mmol/L)",
        dotted_lines=[7, 18], ylim=(0, 0.30), invert_x=False
    )

    print(f"Done. Figures saved to: {FIG_DIR}")
    print(f"CSV files saved to: {CSV_DIR}")
    print(f"N_MERONEN = {N_MERONEN}, N_GHOSHAL = {N_GHOSHAL}, N_LEE = {N_LEE}")

if __name__ == "__main__":
    main()
