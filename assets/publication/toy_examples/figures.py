# %%
import numpy as np
import matplotlib
# matplotlib.use("pgf")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 2. Synchronize fonts with your document
matplotlib.rcParams.update({
    "pgf.texsystem": "xelatex",
    "font.family": "serif",      # Matches default LaTeX Roman fonts
    "font.serif": ["Latin Modern Roman"],
    "font.size": 6,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "text.usetex": True,         # Forces LaTeX compilation
    "pgf.rcfonts": False,        # Prevents Matplotlib overrides
})
# The single column width is: 238.25444pt
# The full page text width is: 494.50888pt
width_inches = 4#238.254 / 72.27
height_inches = width_inches * 1.2  # Clean layout ratio
# %%

# Define the color gradient: light pink to navy blue
colors = ["#FFC0CB", "#000080"]  
pink_to_navy = LinearSegmentedColormap.from_list("PinkToNavy", colors)

# Example usage with a heatmap
data = np.random.rand(10, 10)
plt.imshow(data, cmap=pink_to_navy, interpolation='nearest')
plt.colorbar()
plt.show()

# %%
# 1. Define your exact hex color sequence
# #2E0854: Dark Purple, #FF00FF: Magenta, #FF69B4: Hot Pink, #FFD1B3: Flesh/Peach, #FFFFFF: White
colors = ["#2E0854", "#FF00FF", "#FF69B4", "#FFD1B3", "#FFFFFF"]

# 2. Create the custom colormap
custom_cmap = mcolors.LinearSegmentedColormap.from_list("purple_to_white", colors)

# 3. Register it so you can call it by string name later (optional)
plt.colormaps.register(name="purple_to_white", cmap=custom_cmap)

# --- Visualization Example ---
# Generate sample data
gradient = np.linspace(0, 1, 256).reshape(1, -1)

plt.figure(figsize=(10, 2))
plt.imshow(gradient, aspect='auto', cmap='purple_to_white')
plt.axis('off')
plt.title("Custom Gradient Showcase", fontsize=12, pad=10)
plt.tight_layout()
plt.show()

# %%
# 1. Define the color hexes
# colors = ["#FFFFFF", "#FFD1B3", "#FF69B4", "#DA099B", "#880661", "#2E0854"] # From 0 (White) to 1 (Dark Purple)
colors = ["#FFFFFF", "#fff5f4", "#feeae8", "#fde0dd", "#fbbabd", "#f988ad", "#e84a9b", "#b60982", "#810178", "#49006a"] # From 0 (White) to 1 (Dark Purple)
# colors = ["#FFFFFF", "#feeae8", "#fde0dd", "#fbbabd", "#f988ad", "#e84a9b", "#b60982", "#810178", "#49006a"] # From 0 (White) to 1 (Dark Purple)
colors = ["#ffe4e1", "#eebfbf", "#cc9bb3", "#a3739a", "#7c4c7a", "#605c8a", "#a4b2d6", "#dce3f2"]
colors = ["#605c8a", "#a4b2d6", "#dce3f2", "#ffe4e1", "#eebfbf", "#cc9bb3", "#a3739a", "#7c4c7a"]
custom_cmap_cv = mcolors.LinearSegmentedColormap.from_list("purple_to_white_full", colors)

# By default LinearSegmentedColormap spaces stops evenly by INDEX, not by how
# different they look. The first several colors here are all pale, low-chroma
# pinks, so their perceptual (OKLab) distance is tiny compared to the later,
# more saturated steps -- that's what makes the light end look "crammed" even
# though the index positions are technically uniform. Reparametrizing by
# cumulative OKLab distance makes each step of the colorbar correspond to an
# equal amount of visible change.
def _srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

def _hex_to_oklab(h):
    r, g, b = (int(h.lstrip("#")[i:i+2], 16) / 255 for i in (0, 2, 4))
    r, g, b = _srgb_to_linear(r), _srgb_to_linear(g), _srgb_to_linear(b)
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = l ** (1 / 3), m ** (1 / 3), s ** (1 / 3)
    return np.array([
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    ])

def _perceptual_positions(hex_colors):
    labs = [_hex_to_oklab(c) for c in hex_colors]
    steps = [0.0] + [np.linalg.norm(labs[i] - labs[i - 1]) for i in range(1, len(labs))]
    cum = np.cumsum(steps)
    return cum / cum[-1]

# positions = _perceptual_positions(colors)
# custom_cmap_cv = mcolors.LinearSegmentedColormap.from_list(
#     "purple_to_white_full", list(zip(positions, colors))
# )

# 2. Create sample data that spans from 0 to 1
data = np.linspace(0, 1, 100).reshape(10, 10)

# 3. Plot using vmin and vmax to lock the boundaries
fig, ax = plt.subplots(figsize=(6, 5))
cax = ax.imshow(data, cmap=custom_cmap, vmin=0, vmax=1)

# 4. Add the colorbar with explicit ticks from 0.0 to 1.0
cbar = fig.colorbar(cax, ticks=np.arange(0, 1.1, 0.1))
cbar.ax.set_yticklabels([f"{x:.1f}" for x in np.arange(0, 1.1, 0.1)])

plt.show()

# %%
columns = ['GPC','Gln','Glu','Ins','NAA','PCho','Tau']
rows = []
for l in ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O']:
    rows.append("Model {}".format(l))
CV = [
[1.10, 1.10, 1.90, 1.50, 0.86, 1.70, 0.83],
[0.92, 0.83, 1.40, 0.89, 0.81, 0.90, 2.9],
[0.62, 0.65, 0.66, 0.68, 0.65, 0.68, 0.62],
[0.68, 0.72, 0.78, 0.75, 0.74, 0.71, 0.65],
[0.57, 0.60, 0.56, 0.52, 0.51, 0.59, 0.52],
[0.70, 0.54, 0.55, 0.55, 0.59, 0.64, 0.50],
[0.56, 0.53, 0.50 ,0.51, 0.59, 0.59, 0.45],
[0.55, 0.57, 0.51, 0.53, 0.58, 0.61, 0.46],
[0.56, 0.57, 0.49, 0.67, 0.67, 0.60, 0.49],
[0.67, 0.56, 0.64, 0.60, 0.66, 0.62, 0.49],
[0.59, 4.10, 1.80, 2.00, 0.63, 0.59, 0.47],
[0.57, 0.62, 0.51, 0.51, 0.59, 0.58, 0.49],
[0.61, 0.68, 0.54, 0.54, 0.62, 0.57, 0.49],
[0.84, 1.80, 5.20, 0.62, 0.67, 0.81, 0.71],
[0.51, 0.50, 0.64, 0.81, 0.39, 0.46, 0.50]]

MAPE = [
[0.75, 0.54, 0.94, 0.54, 0.44, 0.49, 0.39],
[0.42, 0.81, 2.10, 0.78, 0.39, 0.29, 1.10],
[0.46, 0.43, 0.46, 0.36, 0.32, 0.34, 0.34],
[0.46, 0.54, 0.50, 0.43, 0.35, 0.40, 0.43],
[0.24, 0.20, 0.33, 0.28, 0.14, 0.13, 0.24],
[0.32, 0.18, 0.30, 0.28, 0.24, 0.15, 0.21],
[0.23, 0.18, 0.26, 0.20, 0.14, 0.12, 0.22],
[0.23, 0.17, 0.22, 0.21, 0.12, 0.12, 0.19],
[0.28, 0.22, 0.25, 0.36, 0.20, 0.16, 0.25],
[0.31, 0.24, 0.35, 0.29, 0.17, 0.18, 0.20],
[0.20, 1.20, 0.53, 0.86, 0.14, 0.12, 0.21],
[0.23, 0.17, 0.25, 0.21, 0.13, 0.12, 0.22],
[0.22, 0.19, 0.20, 0.23, 0.13, 0.13, 0.20],
[0.45, 3.10, 3.60, 0.57, 0.32, 0.27, 0.66],
[0.86, 0.48, 0.80, 0.47, 1.20, 0.63, 0.49]]

# %%
# Define colors from 0 (Red) to 1 (Indigo)
colors = ["#FF0000", "#f988ad", "#FF007F", "#8B008B", "#4B0082"]  # Red -> Pink -> Purple -> Indigo
cmap_name = "red_to_indigo"

# Create the colormap
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

# --- Visualizing the Colormap ---
gradient = np.linspace(0, 1, 256).reshape(1, -1)

# fig, ax = plt.subplots(figsize=(6, 1))
fig, ax = plt.subplots(figsize=(width_inches, height_inches))
fig.subplots_adjust(bottom=0.5)

im = ax.imshow(gradient, aspect='auto', cmap=custom_cmap)
ax.set_title("Colormap: Red (0) to Indigo (1)")
ax.set_xlabel("Value")
ax.set_yticks([])
# fig.colorbar(im, orientation='horizontal', cax=ax)

plt.show()
# %%
CV = np.array(CV)

# fig, ax = plt.subplots(figsize=(len(columns) * 1.2, len(rows) * 1.0))
fig, ax = plt.subplots(figsize=(width_inches, height_inches))

# im = ax.imshow(CV, cmap=custom_cmap_cv, vmin=0, vmax=1)
# im = ax.imshow(CV, cmap="PuOr", vmin=0, vmax=1)
im = ax.imshow(CV, cmap="coolwarm", vmin=0, vmax=1)

ax.set_xticks(np.arange(len(columns)))
ax.set_xticklabels(columns)
ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")

ax.set_yticks(np.arange(len(rows)))
ax.set_yticklabels(rows)

for i in range(CV.shape[0]):
    for j in range(CV.shape[1]):
        color="black" if (CV[i][j]>0.3 and CV[i][j]<0.8) else "white"
        ax.text(j, i, f"{CV[i][j]:.2f}", ha="center", va="center", color=color)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(im, cax=cax)
# cbar.set_label("Value")

ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.0)#5)
ax.tick_params(which="minor", bottom=False, left=False)
plt.tight_layout()

# plt.savefig("/home/john/Documents/Repositories/MRSValidate/ignore/heatmap_coefficients_of_variation.pgf", bbox_inches="tight")
plt.savefig("/home/john/Documents/Repositories/MRSValidate/ignore/heatmap_coefficients_of_variation.pdf", bbox_inches="tight")

plt.show()

 # %%

MAPE = np.array(MAPE)

# fig, ax = plt.subplots(figsize=(len(columns) * 1.2, len(rows) * 1.0))
fig, ax = plt.subplots(figsize=(width_inches, height_inches))

# # im = ax.imshow(CV, cmap=custom_cmap_cv, vmin=0, vmax=1)
# im = ax.imshow(MAPE, cmap="RdYlGn_r", vmin=0.0, vmax=1)
im = ax.imshow(MAPE, cmap="coolwarm", vmin=0.0, vmax=1)
# im = ax.imshow(CV, cmap="GnYlRd", vmin=0, vmax=1)

ax.set_xticks(np.arange(len(columns)))
ax.set_xticklabels(columns)
ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")

ax.set_yticks(np.arange(len(rows)))
ax.set_yticklabels(rows)

for i in range(MAPE.shape[0]):
    for j in range(MAPE.shape[1]):
        color="black" if MAPE[i][j]<0.85 else "white"
        ax.text(j, i, f"{MAPE[i][j]:.2f}", ha="center", va="center", color=color)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(im, cax=cax)
# cbar.set_label("Value")

ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.0)#5)
ax.tick_params(which="minor", bottom=False, left=False)

plt.tight_layout()

# plt.savefig("/home/john/Documents/Repositories/MRSValidate/ignore/heatmap_MAPE.pgf", bbox_inches="tight")
plt.savefig("/home/john/Documents/Repositories/MRSValidate/ignore/heatmap_MAPE.pdf", bbox_inches="tight")
plt.show()
# %%
