"""
diagnostic_abody_gmm.py
========================
Diagnostic to determine how many GMM components are needed to capture
the tails of the body-frame A_body = R^T A R distribution.

For each A_body component (9 total), fits GMMs with n_components in
COMP_RANGE and plots:
  - The empirical KDE
  - The GMM fit for each n_components
  - L2 error vs n_components curve

This tells you the minimum number of components needed before propagating
through M is meaningful — i.e. before you can attribute any remaining
mismatch in the Frobenius norm plots to the norm nonlinearity rather than
GMM approximation quality.

Outputs
-------
  abody_diagnostic/abody_components.png   -- 3x3 grid, all A_body components
  abody_diagnostic/abody_l2_curves.png    -- L2 vs n_components per component
  abody_diagnostic/abody_l2_table.csv     -- L2 errors for all combinations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import os
import time
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture

from jeffery4_2  import Ellipsoid
from orientation_2 import integrate_orientation

# ===========================================================================
# CONFIGURATION
# ===========================================================================

CSV_PATH   = "grad_u.csv"
OUTPUT_DIR = "abody_diagnostic"

a  = np.array([2.0, 1.0, 1.0])
mu = 1.0

# Range of GMM components to try
COMP_RANGE = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 50, 100]

# How many points on the KDE grid
N_KDE_PTS = 600

os.makedirs(OUTPUT_DIR, exist_ok=True)

A_LABELS = [
    [r"$A^{body}_{11}$", r"$A^{body}_{12}$", r"$A^{body}_{13}$"],
    [r"$A^{body}_{21}$", r"$A^{body}_{22}$", r"$A^{body}_{23}$"],
    [r"$A^{body}_{31}$", r"$A^{body}_{32}$", r"$A^{body}_{33}$"],
]

# ===========================================================================
# LOAD AND ROTATE
# ===========================================================================

print("Loading grad_u.csv...")
df = pd.read_csv(CSV_PATH)
t  = df["time"].values
T  = len(t)

A_series = np.zeros((T, 3, 3))
for i in range(3):
    for j in range(3):
        A_series[:, i, j] = df[f"A{i+1}{j+1}"].values

print(f"  {T} timesteps")

print("Integrating orientation ODE...")
t0 = time.perf_counter()
R_history, *_ = integrate_orientation(a, A_series, t)
print(f"  Done in {time.perf_counter()-t0:.2f}s")

# Rotate into body frame
tmp        = np.einsum('tji,tjk->tik', R_history, A_series)
A_body_ts  = np.einsum('tij,tjk->tik', tmp, R_history)
vec_A_body = A_body_ts.reshape(T, 9)   # (T, 9)

# ===========================================================================
# FIT GMMs WITH INCREASING COMPONENTS AND COMPUTE L2 ERRORS
# ===========================================================================

print(f"\nFitting GMMs with n_components in {COMP_RANGE}...")

# Store results: l2[comp_idx, n_comp_idx]
l2_table  = np.zeros((9, len(COMP_RANGE)))
kde_cache = {}   # comp_idx -> (xs, emp_pdf)
gmm_cache = {}   # (comp_idx, n_comp_idx) -> gmm_pdf on xs

for comp_idx in range(9):
    data = vec_A_body[:, comp_idx]
    std  = data.std()

    if std < 1e-10:
        print(f"  Component {comp_idx}: constant, skipping")
        for nc_idx in range(len(COMP_RANGE)):
            l2_table[comp_idx, nc_idx] = 0.0
        kde_cache[comp_idx] = None
        continue

    # Empirical KDE
    spread = data.max() - data.min()
    xs     = np.linspace(data.min() - 0.2*spread,
                         data.max() + 0.2*spread, N_KDE_PTS)
    kde    = gaussian_kde(data)
    emp    = kde(xs)
    kde_cache[comp_idx] = (xs, emp)

    for nc_idx, n_comp in enumerate(COMP_RANGE):
        gmm = GaussianMixture(n_components=n_comp, random_state=0,
                              n_init=3)
        gmm.fit(data.reshape(-1, 1))
        gmm_pdf = np.exp(gmm.score_samples(xs.reshape(-1, 1)))
        l2      = np.sqrt(np.trapezoid((emp - gmm_pdf)**2, xs))
        l2_table[comp_idx, nc_idx] = l2
        gmm_cache[(comp_idx, nc_idx)] = gmm_pdf

    best_nc  = COMP_RANGE[np.argmin(l2_table[comp_idx])]
    best_l2  = l2_table[comp_idx].min()
    i, j     = comp_idx // 3, comp_idx % 3
    print(f"  A_body[{i},{j}]: best n_comp={best_nc}  L2={best_l2:.5f}")

# ===========================================================================
# PLOT 1 — 3x3 grid: empirical KDE + all GMM fits per component
# ===========================================================================

# Colour each GMM curve by n_components
cmap   = cm.viridis
colors = [cmap(v) for v in np.linspace(0.1, 0.9, len(COMP_RANGE))]

fig, axes = plt.subplots(3, 3, figsize=(16, 13))
fig.suptitle(
    "Body-frame A_body component distributions\n"
    "Empirical KDE vs GMM fits with increasing n_components",
    fontsize=13,
)

for comp_idx in range(9):
    i, j = comp_idx // 3, comp_idx % 3
    ax   = axes[i, j]

    if kde_cache[comp_idx] is None:
        ax.text(0.5, 0.5, "constant", ha="center", va="center",
                transform=ax.transAxes, color="grey", fontsize=11)
        ax.set_title(A_LABELS[i][j], fontsize=11)
        continue

    xs, emp = kde_cache[comp_idx]
    ax.fill_between(xs, emp, alpha=0.15, color="steelblue")
    ax.plot(xs, emp, color="steelblue", lw=2.0, label="Empirical KDE", zorder=10)

    for nc_idx, n_comp in enumerate(COMP_RANGE):
        gmm_pdf = gmm_cache[(comp_idx, nc_idx)]
        l2      = l2_table[comp_idx, nc_idx]
        ax.plot(xs, gmm_pdf, color=colors[nc_idx], lw=1.2, alpha=0.85,
                label=f"n={n_comp}  L2={l2:.4f}")

    ax.set_title(A_LABELS[i][j], fontsize=11)
    ax.set_xlabel("value", fontsize=8)
    ax.set_ylabel("density", fontsize=8)
    ax.tick_params(labelsize=7)

    if comp_idx == 0:
        ax.legend(fontsize=6.5, loc="upper right")

# Shared colourbar for n_components
sm = cm.ScalarMappable(
    cmap=cmap,
    norm=plt.Normalize(vmin=COMP_RANGE[0], vmax=COMP_RANGE[-1])
)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02)
cbar.set_label("n_components", fontsize=10)
cbar.set_ticks(COMP_RANGE)

plt.tight_layout(rect=[0, 0, 0.92, 1])
fname = os.path.join(OUTPUT_DIR, "abody_components.png")
plt.savefig(fname, dpi=150)
plt.close(fig)
print(f"\nSaved {fname}")

# ===========================================================================
# PLOT 2 — L2 error vs n_components, one line per A_body component
# ===========================================================================

fig, ax = plt.subplots(figsize=(9, 6))

cmap2  = cm.tab10
colors2 = [cmap2(k/9) for k in range(9)]

for comp_idx in range(9):
    i, j = comp_idx // 3, comp_idx % 3
    if kde_cache[comp_idx] is None:
        continue
    label = f"A_body[{i},{j}]"
    ax.plot(COMP_RANGE, l2_table[comp_idx], "o-",
            color=colors2[comp_idx], label=label, lw=1.5, ms=5)

ax.set_xlabel("n_components", fontsize=12)
ax.set_ylabel("L2 error (GMM vs empirical KDE)", fontsize=12)
ax.set_title(
    "GMM approximation quality vs n_components\n"
    "for each body-frame A_body component\n"
    "(plateau = sufficient components; further increase gives no benefit)",
    fontsize=11,
)
ax.set_xticks(COMP_RANGE)
ax.legend(fontsize=9, ncol=3)
ax.grid(True, alpha=0.3)

# Mark the elbow: where L2 drops below 10% of its n=1 value for each component
for comp_idx in range(9):
    if kde_cache[comp_idx] is None:
        continue
    l2s      = l2_table[comp_idx]
    threshold = 0.1 * l2s[0]
    elbows    = np.where(l2s < threshold)[0]
    if len(elbows):
        elbow_nc = COMP_RANGE[elbows[0]]
        i, j     = comp_idx // 3, comp_idx % 3
        ax.axvline(elbow_nc, color=colors2[comp_idx],
                   lw=0.5, ls=":", alpha=0.5)

plt.tight_layout()
fname = os.path.join(OUTPUT_DIR, "abody_l2_curves.png")
plt.savefig(fname, dpi=150)
plt.close(fig)
print(f"Saved {fname}")

# ===========================================================================
# PLOT 3 — Heatmap: L2 error [component x n_components]
# ===========================================================================

comp_names = [f"A_body[{k//3},{k%3}]" for k in range(9)]
non_const  = [k for k in range(9) if kde_cache[k] is not None]

fig, ax = plt.subplots(figsize=(10, 5))
data_hm  = l2_table[non_const, :]
im = ax.imshow(data_hm, aspect="auto", cmap="YlOrRd",
               vmin=0, vmax=data_hm.max())
ax.set_xticks(range(len(COMP_RANGE)))
ax.set_xticklabels(COMP_RANGE)
ax.set_yticks(range(len(non_const)))
ax.set_yticklabels([comp_names[k] for k in non_const])
ax.set_xlabel("n_components")
ax.set_ylabel("A_body component")
ax.set_title("L2 error heatmap  (lighter = better fit)\n"
             "Read column-wise to find where each row plateaus")
plt.colorbar(im, ax=ax, label="L2 error")

for r, comp_idx in enumerate(non_const):
    for c in range(len(COMP_RANGE)):
        col = "w" if data_hm[r, c] > 0.6 * data_hm.max() else "k"
        ax.text(c, r, f"{data_hm[r,c]:.3f}", ha="center", va="center",
                fontsize=7, color=col)

plt.tight_layout()
fname = os.path.join(OUTPUT_DIR, "abody_l2_heatmap.png")
plt.savefig(fname, dpi=150)
plt.close(fig)
print(f"Saved {fname}")

# ===========================================================================
# SAVE L2 TABLE
# ===========================================================================

df_l2 = pd.DataFrame(
    l2_table,
    index   = comp_names,
    columns = [f"n_comp={n}" for n in COMP_RANGE],
)
df_l2.to_csv(os.path.join(OUTPUT_DIR, "abody_l2_table.csv"))
print(f"Saved {OUTPUT_DIR}/abody_l2_table.csv")

# ===========================================================================
# SUMMARY — recommended minimum n_components
# ===========================================================================

print("\n" + "="*55)
print("Recommended minimum n_components per A_body component")
print("(first n where L2 < 10% of n=1 L2):")
print("="*55)
recommended = {}
for comp_idx in range(9):
    i, j = comp_idx // 3, comp_idx % 3
    if kde_cache[comp_idx] is None:
        print(f"  A_body[{i},{j}]: constant — skip")
        continue
    l2s       = l2_table[comp_idx]
    threshold = 0.1 * l2s[0]
    elbows    = np.where(l2s < threshold)[0]
    if len(elbows):
        rec = COMP_RANGE[elbows[0]]
        print(f"  A_body[{i},{j}]: n_comp >= {rec}  "
              f"(L2={l2s[elbows[0]]:.5f})")
    else:
        rec = COMP_RANGE[-1]
        print(f"  A_body[{i},{j}]: >= {rec} still not plateaued  "
              f"(L2={l2s[-1]:.5f}) — increase COMP_RANGE")
    recommended[f"A_body[{i},{j}]"] = rec

overall = max(recommended.values())
print(f"\n  => Use n_components >= {overall} in body-frame GMM fitting")
print(f"     to ensure tail quality before attributing residual")
print(f"     Frobenius norm mismatch to the norm nonlinearity.")
print("="*55)