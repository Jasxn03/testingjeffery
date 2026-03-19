"""
relationship_A_sigma.py
=======================
Computes the relationship between the velocity gradient A and stress tensor σ
using the TRUE PAIRED samples saved by stress_pdf.py (paired_samples.npz).

Because every row index i corresponds to the same Monte Carlo draw of A that
produced σ, the cross-covariance and mutual information reflect the actual
joint distribution — not two independently re-sampled marginals.

Inputs
------
  paired_samples.npz          — saved by stress_pdf.py
      vec_A             (N, 9)   — A samples, row-major
      sigma_pt1..pt3    (N, 9)   — matched σ samples per surface point
      surface_point_coords (P,3) — coordinates
  gmm_coefficients.csv        — for axis labels / moment reference
  stress_gmm_coefficients.csv — for axis labels / moment reference

Outputs
-------
  cross_covariance.csv         — Cov(A_ij, σ_kl) per surface point
  mutual_information.csv       — I(A_ij; σ_kl) in bits per surface point
  relationship_plots/          — heatmap figures (one per method per point)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.stats import gaussian_kde

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

PAIRED_NPZ    = "paired_samples.npz"
A_GMM_CSV     = "gmm_coefficients.csv"
SIGMA_GMM_CSV = "stress_gmm_coefficients.csv"
OUTPUT_DIR    = "relationship_plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# LOAD PAIRED SAMPLES
# ═══════════════════════════════════════════════════════════════════════════

print("Loading paired samples...")
data  = np.load(PAIRED_NPZ)
vec_A = data["vec_A"]                        # (N, 9)
coords = data["surface_point_coords"]        # (P, 3)
N, _  = vec_A.shape
n_pts = coords.shape[0]

sigma_pts = []
for k in range(1, n_pts + 1):
    sigma_pts.append(data[f"sigma_pt{k}"])   # (N, 9)

print(f"  N = {N} paired samples, {n_pts} surface points")

# Axis labels — read component names from CSVs in the order they appear
df_a = pd.read_csv(A_GMM_CSV)
df_s = pd.read_csv(SIGMA_GMM_CSV)

A_LABELS     = sorted(df_a["component"].unique())   # ['A_00',..'A_22']
SIGMA_LABELS = sorted(df_s["component"].unique())   # ['sigma_00'..'sigma_22']

n_a = len(A_LABELS)
n_s = len(SIGMA_LABELS)

# ═══════════════════════════════════════════════════════════════════════════
# HELPER — KDE entropy estimators
# ═══════════════════════════════════════════════════════════════════════════

def kde_entropy_1d(x):
    if x.std() < 1e-10:
        return 0.0
    kde = gaussian_kde(x)
    lo, hi = x.min() - 3*x.std(), x.max() + 3*x.std()
    xs = np.linspace(lo, hi, 600)
    px = np.clip(kde(xs), 1e-300, None)
    return float(-np.trapezoid(px * np.log2(px), xs))

def kde_entropy_2d(x, y):
    if x.std() < 1e-10 or y.std() < 1e-10:
        return 0.0
    try:
        kde = gaussian_kde(np.vstack([x, y]))
        lx, hx = x.min()-2*x.std(), x.max()+2*x.std()
        ly, hy = y.min()-2*y.std(), y.max()+2*y.std()
        xs = np.linspace(lx, hx, 60)
        ys = np.linspace(ly, hy, 60)
        XX, YY = np.meshgrid(xs, ys)
        pxy = np.clip(
            kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(60, 60),
            1e-300, None
        )
        return float(-np.sum(pxy * np.log2(pxy)) * (xs[1]-xs[0]) * (ys[1]-ys[0]))
    except Exception:
        return 0.0

# ═══════════════════════════════════════════════════════════════════════════
# LOOP OVER SURFACE POINTS
# ═══════════════════════════════════════════════════════════════════════════

cov_rows = []
mi_rows  = []

# Pre-compute marginal entropies of A (shared across all surface points)
print("\nPre-computing A marginal entropies...")
H_A = np.array([kde_entropy_1d(vec_A[:, q]) for q in range(n_a)])

for pt_idx in range(n_pts):
    x_pt      = coords[pt_idx]
    coord_str = f"({x_pt[0]:.3f}, {x_pt[1]:.3f}, {x_pt[2]:.3f})"
    vec_sigma = sigma_pts[pt_idx]   # (N, 9)  — paired with vec_A row-for-row

    # ── Method 2: Cross-covariance ─────────────────────────────────────────

    print(f"\n── Cross-covariance  point {pt_idx+1}  {coord_str} ──")

    joint  = np.vstack([vec_A.T, vec_sigma.T])   # (n_a + n_s, N)
    C_full = np.cov(joint)                        # (n_a+n_s, n_a+n_s)
    C      = C_full[:n_a, n_a:]                   # (n_a, n_s)

    fig, ax = plt.subplots(figsize=(10, 7))
    vmax = np.abs(C).max() or 1.0
    im   = ax.imshow(C, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(n_s)); ax.set_xticklabels(SIGMA_LABELS, rotation=45, ha="right")
    ax.set_yticks(range(n_a)); ax.set_yticklabels(A_LABELS)
    ax.set_xlabel("stress component  σ_kl")
    ax.set_ylabel("velocity gradient  A_ij")
    ax.set_title(
        f"Cov(A_ij,  σ_kl) — paired samples\n"
        f"surface point {pt_idx+1}  {coord_str}",
        fontsize=12,
    )
    plt.colorbar(im, ax=ax, label="covariance")
    for r in range(n_a):
        for c in range(n_s):
            col = "w" if abs(C[r, c]) > 0.6 * vmax else "k"
            ax.text(c, r, f"{C[r,c]:.3f}", ha="center", va="center",
                    fontsize=7, color=col)
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f"crosscov_point{pt_idx+1}.png")
    plt.savefig(fname, dpi=150); plt.close(fig)
    print(f"  Saved {fname}")

    for r in range(n_a):
        for c in range(n_s):
            cov_rows.append({
                "surface_point":   pt_idx + 1,
                "coordinates":     coord_str,
                "A_component":     A_LABELS[r],
                "sigma_component": SIGMA_LABELS[c],
                "covariance":      C[r, c],
            })

    # ── Method 3: Mutual information ───────────────────────────────────────

    print(f"\n── Mutual information  point {pt_idx+1}  {coord_str} ──")

    MI = np.zeros((n_a, n_s))
    for q in range(n_a):
        if H_A[q] < 1e-8:
            continue
        for p in range(n_s):
            H_s      = kde_entropy_1d(vec_sigma[:, p])
            H_js     = kde_entropy_2d(vec_A[:, q], vec_sigma[:, p])
            MI[q, p] = max(0.0, H_A[q] + H_s - H_js)
        if q % 3 == 0:
            print(f"  A row {q//3} done...", flush=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(MI, cmap="YlOrRd", aspect="auto", vmin=0)
    ax.set_xticks(range(n_s)); ax.set_xticklabels(SIGMA_LABELS, rotation=45, ha="right")
    ax.set_yticks(range(n_a)); ax.set_yticklabels(A_LABELS)
    ax.set_xlabel("stress component  σ_kl")
    ax.set_ylabel("velocity gradient  A_ij")
    ax.set_title(
        f"I(A_ij ; σ_kl)  [bits] — paired samples\n"
        f"surface point {pt_idx+1}  {coord_str}",
        fontsize=12,
    )
    plt.colorbar(im, ax=ax, label="MI (bits)")
    for r in range(n_a):
        for c in range(n_s):
            col = "w" if MI[r, c] > 0.7 * MI.max() else "k"
            ax.text(c, r, f"{MI[r,c]:.2f}", ha="center", va="center",
                    fontsize=7, color=col)
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f"MI_point{pt_idx+1}.png")
    plt.savefig(fname, dpi=150); plt.close(fig)
    print(f"  Saved {fname}")

    # Top-10 pairs
    flat_idx = np.argsort(MI.ravel())[::-1][:10]
    print(f"\n  Top-10 (A_ij, σ_kl) pairs by MI:")
    print(f"  {'A_ij':<12} {'σ_kl':<14} {'MI (bits)':>10}")
    for idx in flat_idx:
        r, c = divmod(int(idx), n_s)
        print(f"  {A_LABELS[r]:<12} {SIGMA_LABELS[c]:<14} {MI[r,c]:>10.4f}")

    for r in range(n_a):
        for c in range(n_s):
            mi_rows.append({
                "surface_point":    pt_idx + 1,
                "coordinates":      coord_str,
                "A_component":      A_LABELS[r],
                "sigma_component":  SIGMA_LABELS[c],
                "mutual_info_bits": MI[r, c],
            })

# ═══════════════════════════════════════════════════════════════════════════
# SAVE CSVs
# ═══════════════════════════════════════════════════════════════════════════

pd.DataFrame(cov_rows).to_csv("cross_covariance.csv",  index=False)
pd.DataFrame(mi_rows ).to_csv("mutual_information.csv", index=False)

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("Done. Outputs:")
print("  cross_covariance.csv   — Cov(A_ij, σ_kl) per surface point")
print("  mutual_information.csv — I(A_ij; σ_kl) in bits per surface point")
print(f"  {OUTPUT_DIR}/          — heatmap figures")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")