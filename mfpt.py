"""
mfpt_stress.py
==============
Computes Mean First Passage Time (MFPT) PDFs for the Frobenius norm of the
stress tensor at each of three surface points on the ellipsoid.

Pipeline
--------
  1. Load grad_u.csv  →  time series A(t)
  2. Integrate orientation ODE  →  R(t)  (using existing orientation module)
  3. Build 9×9 stress transfer matrices once per surface point
     (fast_transfer approach:  vec_sigma = M_sigma @ vec(A_body))
  4. Compute stress Frobenius norm time series  ||sigma(t)||_F  at each point
  5. For each surface point and each threshold c:
       - Find all times t where ||sigma(t)||_F first exceeds c after
         previously being below c  (upcrossings of |x| > c)
       - Record the waiting times between successive such passages
  6. Fit a GMM to the waiting-time distribution at each threshold
  7. Plot: one figure per surface point, subplots for each threshold
  8. Save GMM parameters to  mfpt_gmm.csv

Inputs
------
  grad_u.csv            columns: time, A11, A12, A13, A21, A22, A23, A31, A32, A33

Configuration (edit below)
--------------------------
  THRESHOLDS   : list of |sigma|_F values to use as passage thresholds
  a, mu        : ellipsoid semi-axes and fluid viscosity
  surface_points : body-frame points at which to evaluate stress
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

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION  —  edit these
# ═══════════════════════════════════════════════════════════════════════════

CSV_PATH   = "grad_u.csv"
OUTPUT_DIR = "mfpt_plots"
MFPT_CSV   = "mfpt_gmm.csv"

a  = np.array([2.0, 1.0, 1.0])
mu = 1.0

surface_points = [
    np.array([2.0, 0.0, 0.0]),   # tip along x1
    np.array([0.0, 1.0, 0.0]),   # tip along x2
    np.array([0.0, 0.0, 1.0]),   # tip along x3
]

# Thresholds for |sigma|_F — set these based on your data range.
# Run once with THRESHOLDS = [] to print quantiles, then fill in.
THRESHOLDS  = [[10.9252, 12.5103, 14.3799, 16.9575, 21.657],
               [ 5.6187,  6.5587,  7.6369,  9.0286, 11.6326],
               [ 5.3469, 6.1194,  7.1296,  8.5494, 11.7394]]          
N_QUANTILES = 5           # used to auto-suggest thresholds if THRESHOLDS=[]
N_GMM_COMP  = 6           # GMM components for waiting-time distribution

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — Load grad_u time series
# ═══════════════════════════════════════════════════════════════════════════

print("Loading grad_u.csv...")
df   = pd.read_csv(CSV_PATH)
t    = df["time"].values                         # (T,)
T    = len(t)
dt   = np.diff(t)                                # (T-1,)  non-uniform ok

A_series = np.zeros((T, 3, 3))
for i in range(3):
    for j in range(3):
        col = f"A{i+1}{j+1}"
        A_series[:, i, j] = df[col].values

print(f"  Loaded {T} timesteps,  t ∈ [{t[0]:.4g}, {t[-1]:.4g}]")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — Integrate orientation ODE  →  R(t)
# ═══════════════════════════════════════════════════════════════════════════

print("Integrating orientation ODE...")
t0 = time.perf_counter()
R_history, *_ = integrate_orientation(a, A_series, t)   # (T, 3, 3)
print(f"  Done in {time.perf_counter()-t0:.2f}s")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — Build 9×9 stress transfer matrices (once per surface point)
# ═══════════════════════════════════════════════════════════════════════════

def build_stress_transfer_matrix(ellipsoid_class, a, mu, x_surface):
    """
    M_sigma (9×9) such that vec(sigma) = M_sigma @ vec(A_body).
    Built by probing with each of the 9 unit basis velocity gradients.
    """
    x   = np.array(x_surface, dtype=float)
    ell = ellipsoid_class(a, np.zeros((3, 3)), mu=mu)
    ell.use_surface_mode()
    M   = np.zeros((9, 9))
    for q in range(9):
        A_basis            = np.zeros((3, 3))
        A_basis[q//3, q%3] = 1.0
        ell.set_strain(0.5 * (A_basis + A_basis.T))
        ell.set_coefs()
        M[:, q] = np.array(ell.sigma(x)).ravel()
    return M

print("Building stress transfer matrices...")
t0 = time.perf_counter()
M_list = []
for pt_idx, x_pt in enumerate(surface_points):
    print(f"  Point {pt_idx+1}: x = {x_pt} ...", end=" ", flush=True)
    M_list.append(build_stress_transfer_matrix(Ellipsoid, a, mu, x_pt))
    print("done.")
print(f"  Built in {time.perf_counter()-t0:.2f}s")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — Compute stress Frobenius norm time series at each surface point
#
#   A_body(t) = R(t)^T @ A(t) @ R(t)
#   vec_sigma(t) = M_sigma @ vec(A_body(t))      (9,)
#   ||sigma(t)||_F = ||vec_sigma(t)||_2
# ═══════════════════════════════════════════════════════════════════════════

print("Computing stress Frobenius norm time series (vectorised)...")
t0 = time.perf_counter()

# Rotate all A into body frame at once:  A_body[t] = R[t]^T @ A[t] @ R[t]
tmp    = np.einsum('tji,tjk->tik', R_history, A_series)   # R^T @ A
A_body = np.einsum('tij,tjk->tik', tmp,       R_history)  # @ R
vec_A_body = A_body.reshape(T, 9)                          # (T, 9)

# Frobenius norm per surface point:  (T, 9) @ (9, 9).T  →  (T, 9)  →  norm
frob_list = []
for pt_idx, M in enumerate(M_list):
    vec_sigma = vec_A_body @ M.T          # (T, 9)
    frob      = np.linalg.norm(vec_sigma, axis=1)   # (T,)
    frob_list.append(frob)
    print(f"  Point {pt_idx+1}: ||sigma||_F  min={frob.min():.4f}  "
          f"mean={frob.mean():.4f}  max={frob.max():.4f}")

print(f"  Done in {time.perf_counter()-t0:.4f}s")

# ── Auto-suggest thresholds if none provided ──────────────────────────────
if not THRESHOLDS:
    print("\nNo THRESHOLDS set. Suggested values based on data quantiles:")
    for pt_idx, frob in enumerate(frob_list):
        qs = np.quantile(frob, np.linspace(0.5, 0.95, N_QUANTILES))
        print(f"  Point {pt_idx+1}: {np.array2string(qs, precision=4)}")
    print("\nSet THRESHOLDS in the configuration block and re-run.")
    raise SystemExit(0)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — Extract first-passage waiting times for each threshold
#
# Definition: |sigma(t)|_F exceeds threshold c.
# A "passage event" occurs at the first timestep where the series crosses
# from below c to above c (upcrossing of the level c).
# The waiting time is the elapsed time since the previous upcrossing
# (or since t=0 for the first event).
# ═══════════════════════════════════════════════════════════════════════════

def first_passage_times(frob, t, threshold):
    """
    Return array of waiting times between successive upcrossings of
    |sigma|_F = threshold.

    A new passage clock starts each time the process drops back below
    the threshold and then crosses it again.
    """
    above      = frob > threshold          # boolean mask (T,)
    # upcrossing: False at t-1, True at t
    crossings  = np.where(~above[:-1] & above[1:])[0] + 1   # indices in [1, T-1]

    if len(crossings) < 2:
        return np.array([])                # not enough events

    # Waiting times = time between successive upcrossings
    crossing_times = t[crossings]
    waiting_times  = np.diff(crossing_times)
    return waiting_times

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6 & 7 — Fit GMM, plot, save
# ═══════════════════════════════════════════════════════════════════════════

gmm_rows = []
 
for pt_idx, (frob, x_pt) in enumerate(zip(frob_list, surface_points)):
    coord_str      = f"({x_pt[0]:.1f}, {x_pt[1]:.1f}, {x_pt[2]:.1f})"
    pt_thresholds  = THRESHOLDS[pt_idx]
    n_thresh       = len(pt_thresholds)
    ncols          = min(3, n_thresh)
    nrows          = int(np.ceil(n_thresh / ncols))
 
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5*ncols, 4*nrows),
                             squeeze=False)
    fig.suptitle(
        f"MFPT distribution -- surface point {pt_idx+1}  x = {coord_str}\n"
        f"Frobenius norm  ||sigma||_F  recurrence times",
        fontsize=13,
    )
 
    for th_idx, thresh in enumerate(pt_thresholds):
        ax  = axes[th_idx // ncols][th_idx % ncols]
        wts = first_passage_times(frob, t, thresh)
 
        if len(wts) < 10:
            ax.text(0.5, 0.5,
                    f"threshold = {thresh}\n< 10 events ({len(wts)} found)\n"
                    f"Try a lower threshold.",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="grey")
            ax.set_title(f"c = {thresh}")
            gmm_rows.append({
                "surface_point":   pt_idx + 1,
                "coordinates":     coord_str,
                "threshold":       thresh,
                "n_events":        len(wts),
                "gmm_component_k": np.nan,
                "weight":          np.nan,
                "mean":            np.nan,
                "variance":        np.nan,
                "mfpt_empirical":  np.nan,
            })
            continue
 
        mfpt_emp = wts.mean()
        print(f"  Point {pt_idx+1}  c={thresh:.4g}: "
              f"{len(wts)} events,  MFPT = {mfpt_emp:.4f}")
 
        # KDE
        spread = wts.max() - wts.min() + 1e-12
        xs     = np.linspace(max(0, wts.min() - 0.05*spread),
                             wts.max() + 0.1*spread, 500)
        kde    = gaussian_kde(wts)
        ax.fill_between(xs, kde(xs), alpha=0.25, color="steelblue")
        ax.plot(xs, kde(xs), color="steelblue", lw=1.5, label="KDE")
 
        # GMM
        n_comp = min(N_GMM_COMP, max(1, len(wts) // 5))
        gmm    = GaussianMixture(n_components=n_comp, random_state=0)
        gmm.fit(wts.reshape(-1, 1))
        ax.plot(xs, np.exp(gmm.score_samples(xs.reshape(-1, 1))),
                "--", color="tomato", lw=1.8, label="GMM")
 
        ax.axvline(mfpt_emp, color="k", lw=1, ls=":",
                   label=f"MFPT = {mfpt_emp:.3f}")
        ax.set_title(f"c = {thresh}  ({len(wts)} events)")
        ax.set_xlabel("waiting time")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
 
        for k in range(n_comp):
            gmm_rows.append({
                "surface_point":   pt_idx + 1,
                "coordinates":     coord_str,
                "threshold":       thresh,
                "n_events":        len(wts),
                "gmm_component_k": k,
                "weight":          gmm.weights_[k],
                "mean":            gmm.means_[k, 0],
                "variance":        gmm.covariances_[k, 0, 0],
                "mfpt_empirical":  mfpt_emp,
            })
 
    # Hide unused subplots
    for idx in range(n_thresh, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
 
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f"mfpt_point{pt_idx+1}.png")
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")
 
    fig_ov, axes_ov = plt.subplots(1, 2, figsize=(13, 5))
    fig_ov.suptitle(
        f"Recurrence time distributions -- all thresholds"
        f"Surface point {pt_idx+1}  x = {coord_str}",
        fontsize=13,
    )
 
    cmap_ov    = plt.cm.plasma
    colours_ov = [cmap_ov(v) for v in np.linspace(0.1, 0.85, n_thresh)]
    ax_kde = axes_ov[0]
    ax_gmm = axes_ov[1]
 
    for th_idx, thresh in enumerate(pt_thresholds):
        wts = first_passage_times(frob, t, thresh)
        if len(wts) < 10:
            continue
        colour   = colours_ov[th_idx]
        mfpt_emp = wts.mean()
        spread   = wts.max() - wts.min() + 1e-12
        xs       = np.linspace(max(0, wts.min() - 0.05*spread),
                               wts.max() + 0.1*spread, 500)
        kde = gaussian_kde(wts)
        ax_kde.plot(xs, kde(xs), color=colour, lw=1.8,
                    label=f"c={thresh}  MFPT={mfpt_emp:.3f}")
        ax_kde.axvline(mfpt_emp, color=colour, lw=0.8, ls=":" )
 
        n_comp = min(N_GMM_COMP, max(1, len(wts) // 5))
        gmm    = GaussianMixture(n_components=n_comp, random_state=0)
        gmm.fit(wts.reshape(-1, 1))
        ax_gmm.plot(xs, np.exp(gmm.score_samples(xs.reshape(-1, 1))),
                    color=colour, lw=1.8,
                    label=f"c={thresh}  MFPT={mfpt_emp:.3f}")
        ax_gmm.axvline(mfpt_emp, color=colour, lw=0.8, ls=":" )
 
    ax_kde.set_xlabel("waiting time"); ax_kde.set_ylabel("density")
    ax_kde.set_title("KDE -- all thresholds overlaid")
    ax_kde.legend(fontsize=8)
    ax_gmm.set_xlabel("waiting time"); ax_gmm.set_ylabel("density")
    ax_gmm.set_title("GMM fit -- all thresholds overlaid")
    ax_gmm.legend(fontsize=8)
 
    sm = plt.cm.ScalarMappable(
        cmap=cmap_ov,
        norm=plt.Normalize(vmin=pt_thresholds[0], vmax=pt_thresholds[-1])
    )
    sm.set_array([])
    fig_ov.colorbar(sm, ax=axes_ov, shrink=0.7, label="threshold  c", pad=0.02)
    plt.tight_layout()
    fname_ov = os.path.join(OUTPUT_DIR, f"mfpt_overlay_point{pt_idx+1}.png")
    plt.savefig(fname_ov, dpi=150)
    plt.close(fig_ov)
    print(f"  Saved {fname_ov}")
    
    # -- Log-scale version: 2 rows (log-x linear-y, log-x log-y) ----------
    fig_log, axes_log = plt.subplots(1, 2, figsize=(13, 9))
    fig_log.suptitle(
        f"Recurrence time distributions -- log scale"
        f"Surface point {pt_idx+1}  x = {coord_str}",
        fontsize=13,
    )
    titles = ["KDE  (log x, linear y)", "GMM  (log x, linear y)"]
    for ax, title in zip(axes_log.ravel(), titles):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("waiting time")
        ax.set_ylabel("density")
        ax.set_xscale("log")
 
    for th_idx, thresh in enumerate(pt_thresholds):
        wts = first_passage_times(frob, t, thresh)
        if len(wts) < 10:
            continue
        colour   = colours_ov[th_idx]
        mfpt_emp = wts.mean()
 
        # Use log-spaced x grid so curves are smooth on log axis
        xlo = max(wts.min() * 0.5, 1e-6)
        xhi = wts.max() * 1.5
        xs_log = np.geomspace(xlo, xhi, 600)
 
        kde     = gaussian_kde(wts)
        kde_pdf = kde(xs_log)
        kde_pdf = np.clip(kde_pdf, 1e-10, None)
 
        n_comp = min(N_GMM_COMP, max(1, len(wts) // 5))
        gmm    = GaussianMixture(n_components=n_comp, random_state=0)
        gmm.fit(wts.reshape(-1, 1))
        gmm_pdf = np.exp(gmm.score_samples(xs_log.reshape(-1, 1)))
        gmm_pdf = np.clip(gmm_pdf, 1e-10, None)
 
        lab = f"c={thresh:.4g}  MFPT={mfpt_emp:.3f}"
 
        for ax, pdf in [(axes_log[0], kde_pdf),
                        (axes_log[1], gmm_pdf)]:
            ax.plot(xs_log, pdf, color=colour, lw=1.6, label=lab)
            ax.axvline(mfpt_emp, color=colour, lw=0.7, ls=":")
 
    for ax in axes_log.ravel():
        ax.legend(fontsize=7)
 
    sm2 = plt.cm.ScalarMappable(
        cmap=cmap_ov,
        norm=plt.Normalize(vmin=pt_thresholds[0], vmax=pt_thresholds[-1])
    )
    sm2.set_array([])
    # fig_log.colorbar(sm2, ax=axes_log, shrink=0.6,
    #                  label="threshold  c", pad=0.02)
    plt.tight_layout()
    fname_log = os.path.join(OUTPUT_DIR, f"mfpt_overlay_log_point{pt_idx+1}.png")
    plt.savefig(fname_log, dpi=150)
    plt.close(fig_log)
    print(f"  Saved {fname_log}")

# -- Summary: MFPT vs threshold, all three points on one axes --------------
fig, ax = plt.subplots(figsize=(8, 5))
colours = ["steelblue", "tomato", "seagreen"]
df_gmm  = pd.DataFrame(gmm_rows)
 
for pt_idx, x_pt in enumerate(surface_points):
    coord_str = f"({x_pt[0]:.1f}, {x_pt[1]:.1f}, {x_pt[2]:.1f})"
    sub = (df_gmm[df_gmm["surface_point"] == pt_idx + 1]
           .drop_duplicates("threshold")
           .dropna(subset=["mfpt_empirical"]))
    ax.plot(sub["threshold"], sub["mfpt_empirical"],
            "o-", color=colours[pt_idx],
            label=f"Point {pt_idx+1}  {coord_str}")
 
ax.set_xlabel("threshold  c")
ax.set_ylabel("MFPT  (time units)")
ax.set_title("Mean first passage time vs threshold")
ax.legend(fontsize=9)
plt.tight_layout()
fname = os.path.join(OUTPUT_DIR, "mfpt_vs_threshold.png")
plt.savefig(fname, dpi=150)
plt.close(fig)
print(f"Saved summary plot -> {fname}")
 
# -- Save CSV --------------------------------------------------------------
pd.DataFrame(gmm_rows, columns=[
    "surface_point", "coordinates", "threshold", "n_events",
    "gmm_component_k", "weight", "mean", "variance", "mfpt_empirical",
]).to_csv(MFPT_CSV, index=False)
print(f"Saved GMM parameters -> {MFPT_CSV}")
 
print("\n" + "="*45)
print("Done. Outputs:")
print(f"  {OUTPUT_DIR}/mfpt_point1..3.png    -- MFPT PDFs per surface point")
print(f"  {OUTPUT_DIR}/mfpt_vs_threshold.png -- MFPT vs threshold summary")
print(f"  {MFPT_CSV}                         -- GMM parameters")
print("="*45)
 