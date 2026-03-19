"""
stress_pdf.py
=============
Pipeline:
  1. Load grad_u.csv  ->  A(t) time series  (the true joint distribution of A)
  2. Integrate orientation ODE  ->  R(t)
  3. Rotate into body frame:  A_body(t) = R(t)^T A(t) R(t)
  4. Build 9x9 stress transfer matrices once per surface point
  5. Compute stress time series:  vec_sigma(t) = M @ vec(A_body(t))
  6. Plot PDFs (KDE + GMM fit) for every stress component at every surface point
  7. Fit GMM to each stress component distribution
  8. Save GMM parameters  ->  stress_gmm_coefficients.csv
  9. Save paired (vec_A_body, vec_sigma) arrays  ->  paired_samples.npz

Why use the time series directly instead of sampling from the A-GMM
--------------------------------------------------------------------
The GMM fitted to A in pdf_CM.py treats each of the 9 components
independently (diagonal joint covariance). The true joint distribution of
A has strong correlations between components — e.g. incompressibility
constrains tr(A) = 0, and vorticity couples off-diagonal pairs.
Sampling from 9 independent marginals produces physically impossible A
tensors and gives the wrong stress distribution.

The time series IS the correct set of samples from the true joint
distribution of A. Using it directly requires no GMM for A at all.

Inputs
------
  grad_u.csv   columns: time, A11, A12, A13, A21, A22, A23, A31, A32, A33

Outputs
-------
  stress_pdfs/stress_pdf_point{k}.png   -- KDE + GMM for each sigma_ij
  stress_gmm_coefficients.csv           -- GMM parameters per component
  paired_samples.npz                    -- paired (vec_A_body, vec_sigma)
                                           for relationship_A_sigma.py
"""

import numpy as np
import matplotlib.pyplot as plt
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
N_GMM_COMP = 6
OUTPUT_DIR = "stress_pdfs"
STRESS_CSV = "stress_gmm_coefficients.csv"

a  = np.array([2.0, 1.0, 1.0])
mu = 1.5

surface_points = [
    np.array([2.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, 0.0, 1.0]),
]

STRESS_LABELS = [
    [r"$\sigma_{11}$", r"$\sigma_{12}$", r"$\sigma_{13}$"],
    [r"$\sigma_{21}$", r"$\sigma_{22}$", r"$\sigma_{23}$"],
    [r"$\sigma_{31}$", r"$\sigma_{32}$", r"$\sigma_{33}$"],
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================================================================
# STEP 1 -- Load time series
# ===========================================================================

print("Loading grad_u.csv...")
df = pd.read_csv(CSV_PATH)
t  = df["time"].values
T  = len(t)

A_series = np.zeros((T, 3, 3))
for i in range(3):
    for j in range(3):
        A_series[:, i, j] = df[f"A{i+1}{j+1}"].values

print(f"  {T} timesteps,  t in [{t[0]:.4g}, {t[-1]:.4g}]")

# ===========================================================================
# STEP 2 -- Integrate orientation ODE  ->  R(t)
# ===========================================================================

print("Integrating orientation ODE...")
t0 = time.perf_counter()
R_history, *_ = integrate_orientation(a, A_series, t)
print(f"  Done in {time.perf_counter()-t0:.2f}s")

# ===========================================================================
# STEP 3 -- Rotate into body frame
#
#   A_body(t) = R(t)^T A(t) R(t)
#
# This is the quantity that enters the stress transfer matrix.
# Using the body-frame A preserves all correlations between components
# exactly as they appear in the data.
# ===========================================================================

print("Rotating A into body frame...")
t0 = time.perf_counter()
tmp        = np.einsum('tji,tjk->tik', R_history, A_series)   # R^T @ A
A_body_ts  = np.einsum('tij,tjk->tik', tmp, R_history)         # @ R
vec_A_body = A_body_ts.reshape(T, 9)                            # (T, 9)
print(f"  Done in {time.perf_counter()-t0:.3f}s")

# ===========================================================================
# STEP 4 -- Build stress transfer matrices (once per surface point)
#
#   vec(sigma) = M_sigma @ vec(A_body)
# ===========================================================================

def build_stress_transfer_matrix(ellipsoid_class, a, mu, x_surface):
    x   = np.array(x_surface, dtype=float)
    n   = x / (a ** 2); n /= np.linalg.norm(n)
    ell = ellipsoid_class(a, np.zeros((3, 3)), mu=mu)
    ell.use_surface_mode()
    M   = np.zeros((9, 9))
    for q in range(9):
        A_basis             = np.zeros((3, 3))
        A_basis[q//3, q%3]  = 1.0
        ell.set_strain(0.5 * (A_basis + A_basis.T))
        ell.set_coefs()
        M[:, q] = np.array(ell.sigma(x)).ravel()
    return M, n

print("Building stress transfer matrices...")
t0 = time.perf_counter()
transfer = []
for pt_idx, x_pt in enumerate(surface_points):
    print(f"  Point {pt_idx+1}: x = {x_pt} ...", end=" ", flush=True)
    M, n_hat = build_stress_transfer_matrix(Ellipsoid, a, mu, x_pt)
    transfer.append({"M": M, "n_hat": n_hat, "x": x_pt})
    print("done.")
print(f"  Built in {time.perf_counter()-t0:.2f}s\n")

# ===========================================================================
# STEP 5 -- Compute stress time series at each surface point
#
#   vec_sigma(t) = M @ vec(A_body(t))      (T, 9) = (T, 9) @ (9, 9).T
#
# T is the number of timesteps — these ARE the samples from the true
# joint distribution of stress, with all correlations intact.
# ===========================================================================

print("Computing stress time series (vectorised matmul)...")
t0 = time.perf_counter()

sigma_ts = []    # list of (T, 3, 3) arrays, one per surface point
for td in transfer:
    vec_sigma = vec_A_body @ td["M"].T          # (T, 9)
    sigma_ts.append(vec_sigma.reshape(T, 3, 3))

print(f"  Done in {time.perf_counter()-t0:.4f}s  ({T} timesteps)\n")

# ===========================================================================
# STEP 6 -- Plot PDFs: KDE + GMM for each sigma_ij at each surface point
# ===========================================================================

for pt_idx, x_pt in enumerate(surface_points):
    coord_str = f"({x_pt[0]:.1f}, {x_pt[1]:.1f}, {x_pt[2]:.1f})"
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    fig.suptitle(
        f"Stress PDF — surface point {pt_idx+1}   x = {coord_str}\n"
        f"({T} timesteps from grad_u.csv)",
        fontsize=13,
    )

    for i in range(3):
        for j in range(3):
            ax   = axes[i, j]
            data = sigma_ts[pt_idx][:, i, j]
            std  = data.std()

            if std < 1e-10:
                ax.text(0.5, 0.5,
                        f"constant\n(value ≈ {data.mean():.3g})",
                        ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="grey")
                ax.set_title(STRESS_LABELS[i][j], fontsize=11)
                ax.set_xlabel("stress value")
                ax.set_ylabel("density")
                continue

            spread = data.max() - data.min()
            xs     = np.linspace(data.min() - 0.1*spread,
                                 data.max() + 0.1*spread, 400)

            kde = gaussian_kde(data)
            ax.fill_between(xs, kde(xs), alpha=0.25, color="steelblue")
            ax.plot(xs, kde(xs), color="steelblue", lw=1.5, label="KDE")

            gmm = GaussianMixture(n_components=N_GMM_COMP, random_state=0,
                                  n_init=3)
            gmm.fit(data.reshape(-1, 1))
            ax.plot(xs, np.exp(gmm.score_samples(xs.reshape(-1, 1))),
                    "--", color="tomato", lw=1.5, label="GMM")

            ax.set_title(STRESS_LABELS[i][j], fontsize=11)
            ax.set_xlabel("stress value")
            ax.set_ylabel("density")
            if i == 0 and j == 0:
                ax.legend(fontsize=8)

    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f"stress_pdf_point{pt_idx+1}.png")
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")

print("All figures saved.\n")

# ===========================================================================
# STEP 7 -- Fit GMM to each stress component & save CSV
# ===========================================================================

rows = []

for pt_idx, x_pt in enumerate(surface_points):
    coord_str = f"({x_pt[0]:.3f}, {x_pt[1]:.3f}, {x_pt[2]:.3f})"
    for i in range(3):
        for j in range(3):
            data       = sigma_ts[pt_idx][:, i, j]
            comp_label = f"sigma_{i}{j}"

            if data.std() < 1e-10:
                rows.append({
                    "surface_point":   pt_idx + 1,
                    "coordinates":     coord_str,
                    "component":       comp_label,
                    "gmm_component_k": 0,
                    "weight":          1.0,
                    "mean":            float(data.mean()),
                    "variance":        0.0,
                })
                continue

            gmm = GaussianMixture(n_components=N_GMM_COMP, random_state=0,
                                  n_init=3)
            gmm.fit(data.reshape(-1, 1))

            for k in range(N_GMM_COMP):
                rows.append({
                    "surface_point":   pt_idx + 1,
                    "coordinates":     coord_str,
                    "component":       comp_label,
                    "gmm_component_k": k,
                    "weight":          gmm.weights_[k],
                    "mean":            gmm.means_[k, 0],
                    "variance":        gmm.covariances_[k, 0, 0],
                })

df_out = pd.DataFrame(rows, columns=[
    "surface_point", "coordinates", "component",
    "gmm_component_k", "weight", "mean", "variance",
])
df_out.to_csv(STRESS_CSV, index=False)
print(f"Saved stress GMM coefficients -> {STRESS_CSV}")

# ===========================================================================
# STEP 8 -- Save paired samples for relationship_A_sigma.py
#
# vec_A_body  : (T, 9)  body-frame A at every timestep
# sigma_pt{k} : (T, 9)  matched stress at surface point k
#
# Row i corresponds to timestep i — pairing is exact.
# ===========================================================================

paired_data = {"vec_A": vec_A_body}
for pt_idx, sig in enumerate(sigma_ts):
    paired_data[f"sigma_pt{pt_idx+1}"] = sig.reshape(T, 9)
paired_data["surface_point_coords"] = np.array(surface_points)

np.savez("paired_samples.npz", **paired_data)
print("Paired samples saved -> paired_samples.npz")
print(f"  Shape: vec_A = {vec_A_body.shape}, "
      f"sigma_pt1 = {sigma_ts[0].reshape(T,9).shape}")