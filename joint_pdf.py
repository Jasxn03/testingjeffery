"""
joint_pdf_mfpt_sigma.py
=======================
For each surface point and each threshold, plots the joint PDF of:

  Option 2: recurrence time  vs  PEAK ||sigma||_F during the excursion
            (maximum stress reached between two successive upcrossings)

  Option 3: recurrence time  vs  MEAN ||sigma||_F during the excursion
            (average stress over the time spent above the threshold)

One figure per surface point, with subplots arranged as:
  rows    = thresholds
  columns = [joint PDF peak | marginals peak | joint PDF mean | marginals mean]

Each joint PDF panel shows:
  - 2D KDE colourmap
  - Marginal histograms on the axes
  - Pearson correlation coefficient
  - Scatter of individual excursion events (semi-transparent)

Inputs
------
  grad_u.csv   -- time series of A(t)
  Same THRESHOLDS, a, mu, surface_points as mfpt_stress.py

Outputs
-------
  joint_pdf/joint_pdf_point{k}.png   -- one figure per surface point
  joint_pdf/joint_stats.csv          -- correlation, mean, std per condition
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import os
import time
from scipy.stats import gaussian_kde, pearsonr, spearmanr

from jeffery4_2  import Ellipsoid
from orientation_2 import integrate_orientation

# ===========================================================================
# CONFIGURATION  -- keep consistent with mfpt_stress.py
# ===========================================================================

CSV_PATH   = "grad_u.csv"
OUTPUT_DIR = "joint_pdf"

a  = np.array([2.0, 1.0, 1.0])
mu = 1.0

surface_points = [
    np.array([2.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, 0.0, 1.0]),
]

# Same per-point threshold lists as mfpt_stress.py
THRESHOLDS  = [[16.9575, 21.657],
               [9.0286, 11.6326],
               [8.5494, 11.7394]]  # reduce number of thresholds so i can actually see the plot, just use 83 and 95th percentile

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================================================================
# LOAD, ROTATE, COMPUTE FROBENIUS NORM TIME SERIES
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

tmp        = np.einsum('tji,tjk->tik', R_history, A_series)
A_body_ts  = np.einsum('tij,tjk->tik', tmp, R_history)
vec_A_body = A_body_ts.reshape(T, 9)

def build_stress_transfer_matrix(ellipsoid_class, a, mu, x_surface):
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

print("Building transfer matrices...")
frob_list = []
for pt_idx, x_pt in enumerate(surface_points):
    print(f"  Point {pt_idx+1} ...", end=" ", flush=True)
    M    = build_stress_transfer_matrix(Ellipsoid, a, mu, x_pt)
    frob = np.linalg.norm(vec_A_body @ M.T, axis=1)
    frob_list.append(frob)
    print(f"done.  mean={frob.mean():.3f}  max={frob.max():.3f}")

# ===========================================================================
# EXCURSION EXTRACTOR
# ===========================================================================

def excursion_stats(frob, t, threshold):
    """
    For each excursion above threshold, return:
      recurrence_time : time since previous upcrossing          (N_events,)
      peak_mag        : max ||sigma||_F during the excursion     (N_events,)
      mean_mag        : mean ||sigma||_F during the excursion    (N_events,)

    An excursion is the contiguous block of timesteps where frob > threshold
    between two successive upcrossings.
    """
    above     = frob > threshold
    # upcrossing indices: transition F->T
    up_idx    = np.where(~above[:-1] &  above[1:])[0] + 1
    # downcrossing indices: transition T->F
    down_idx  = np.where( above[:-1] & ~above[1:])[0] + 1

    if len(up_idx) < 2:
        return np.array([]), np.array([]), np.array([])

    # Align: for each upcrossing find the next downcrossing
    rec_times = []
    peak_mags = []
    mean_mags = []

    crossing_t = t[up_idx]
    rec_times_raw = np.diff(crossing_t)   # recurrence times between upcrossings

    for ev_idx in range(len(up_idx) - 1):
        u = up_idx[ev_idx]

        # Find the downcrossing that follows this upcrossing
        later_downs = down_idx[down_idx > u]
        if len(later_downs) == 0:
            continue
        d = later_downs[0]

        # Excursion is the block frob[u:d]
        excursion = frob[u:d]
        if len(excursion) == 0:
            continue

        rec_times.append(rec_times_raw[ev_idx])
        peak_mags.append(excursion.max())
        mean_mags.append(excursion.mean())

    return (np.array(rec_times),
            np.array(peak_mags),
            np.array(mean_mags))

# ===========================================================================
# JOINT PDF PLOTTING HELPER
# ===========================================================================

def plot_joint_panel(ax_joint, ax_top, ax_right,
                     x_data, y_data, xlabel, ylabel,
                     threshold, cmap="viridis"):
    """
    Plot a 2D KDE joint PDF with marginal histograms.
    ax_joint : main 2D KDE axes
    ax_top   : marginal histogram for x (recurrence time)
    ax_right : marginal histogram for y (stress magnitude)
    """
    if len(x_data) < 10:
        ax_joint.text(0.5, 0.5, f"< 10 events\n({len(x_data)} found)",
                      ha="center", va="center", transform=ax_joint.transAxes,
                      color="grey", fontsize=9)
        return
    
    x_pos = np.clip(x_data, 1e-9, None)
    log_x = np.log10(x_pos)
 
    xlo, xhi = log_x.min() - 0.1, log_x.max() + 0.1
    ypad = 0.05 * (y_data.max() - y_data.min())
    ylo  = y_data.min() - ypad
    yhi  = y_data.max() + ypad

    # 2D KDE
    try:
        kde2d  = gaussian_kde(np.vstack([log_x, y_data]), bw_method="scott")
        xs_k   = np.linspace(xlo, xhi, 100)
        ys_k   = np.linspace(ylo, yhi, 100)
        XX, YY = np.meshgrid(xs_k, ys_k)
        ZZ     = kde2d(np.vstack([XX.ravel(), YY.ravel()])).reshape(100, 100)
 
        # Clip colour scale at 99th percentile so dense spike
        # doesn't wash out structure elsewhere
        vmax = np.percentile(ZZ, 99)
        ax_joint.contourf(XX, YY, ZZ,
                          levels=15, cmap=cmap,
                          vmin=0, vmax=vmax, alpha=0.9)
        ax_joint.contour(XX, YY, ZZ,
                         levels=8, colors="white",
                         linewidths=0.5, alpha=0.6,
                         vmax=vmax)
    except Exception:
        ax_joint.scatter(log_x, y_data, s=4, alpha=0.3, color="steelblue")

    # Pearson and Spearman correlations
    r_p, _ = pearsonr(x_data,  y_data)
    r_s, _ = spearmanr(x_data, y_data)
    ax_joint.text(0.97, 0.97,
                  f"r_P={r_p:.3f}\nr_S={r_s:.3f}",
                  transform=ax_joint.transAxes,
                  ha="right", va="top", fontsize=8,
                  color="white",
                  bbox=dict(boxstyle="round,pad=0.2",
                            fc="black", alpha=0.4))

    # x-ticks as 10^n labels
    tick_log = np.arange(np.floor(xlo), np.ceil(xhi) + 1)
    tick_log = tick_log[(tick_log >= xlo) & (tick_log <= xhi)]
    ax_joint.set_xticks(tick_log)
    ax_joint.set_xticklabels(
        [f"$10^{{{int(v)}}}$" if v != 0 else "1" for v in tick_log],
        fontsize=7)
    ax_joint.set_xlim(xlo, xhi)
    ax_joint.set_ylim(ylo, yhi)
    ax_joint.tick_params(axis="y", labelsize=7)
 
    ax_joint.axhline(threshold, color="tomato", lw=0.8, ls="--", alpha=0.7)
    ax_joint.set_xlabel(xlabel + "  (log scale)", fontsize=8)
    ax_joint.set_ylabel(ylabel, fontsize=8)
 
    # Marginal KDE — top (recurrence time, log-transformed)
    if log_x.std() > 1e-10:
        kde_x = gaussian_kde(log_x)
        xs_m  = np.linspace(xlo, xhi, 300)
        ax_top.fill_between(xs_m, kde_x(xs_m), alpha=0.35, color="steelblue")
        ax_top.plot(xs_m, kde_x(xs_m), color="steelblue", lw=1.2)
    ax_top.set_xlim(xlo, xhi)
    ax_top.axis("off")
 
    # Marginal KDE — right (stress magnitude)
    if y_data.std() > 1e-10:
        kde_y = gaussian_kde(y_data)
        ys_m  = np.linspace(ylo, yhi, 300)
        ax_right.fill_betweenx(ys_m, kde_y(ys_m), alpha=0.35, color="seagreen")
        ax_right.plot(kde_y(ys_m), ys_m, color="seagreen", lw=1.2)
    ax_right.set_ylim(ylo, yhi)
    ax_right.axis("off")

# ===========================================================================
# MAIN LOOP — one figure per surface point
# ===========================================================================

stats_rows = []

for pt_idx, (frob, x_pt) in enumerate(zip(frob_list, surface_points)):
    coord_str     = f"({x_pt[0]:.1f}, {x_pt[1]:.1f}, {x_pt[2]:.1f})"
    pt_thresholds = THRESHOLDS[pt_idx]
    n_thresh      = len(pt_thresholds)

    # Layout: n_thresh rows x 4 panels per row
    # Each panel = [joint_peak | joint_mean]
    # Each joint panel has marginals → use nested GridSpec
    #
    # Per row: 4 sub-columns arranged as:
    #   [top_peak  |  spacer  |  top_mean  |  spacer  ]
    #   [joint_peak | right_p |  joint_mean | right_m ]
    # We use gridspec_kw to control relative sizes

    fig = plt.figure(figsize=(18, 5 * n_thresh))
    fig.suptitle(
        f"Joint PDF: recurrence time × stress magnitude\n"
        f"Surface point {pt_idx+1}  x = {coord_str}",
        fontsize=13, y=1.01,
    )

    outer = gridspec.GridSpec(n_thresh, 2, figure=fig,
                              hspace=0.45, wspace=0.35)

    for th_idx, thresh in enumerate(pt_thresholds):

        rec, peak, mean_mag = excursion_stats(frob, t, thresh)

        print(f"  Point {pt_idx+1}  c={thresh:.4g}: "
              f"{len(rec)} excursions", flush=True)

        for col_idx, (y_data, y_label, panel_title) in enumerate([
            (peak,    f"Peak ||σ||_F",  f"Peak stress  (c={thresh})"),
            (mean_mag, f"Mean ||σ||_F", f"Mean stress  (c={thresh})"),
        ]):
            # Each cell split into a 2x2 mini-grid for joint+marginals
            inner = gridspec.GridSpecFromSubplotSpec(
                2, 2,
                subplot_spec=outer[th_idx, col_idx],
                height_ratios=[1, 4],
                width_ratios=[4, 1],
                hspace=0.05, wspace=0.05,
            )
            ax_top   = fig.add_subplot(inner[0, 0])
            ax_joint = fig.add_subplot(inner[1, 0])
            ax_right = fig.add_subplot(inner[1, 1])
            # Top-right corner blank
            fig.add_subplot(inner[0, 1]).axis("off")

            ax_joint.set_title(panel_title, fontsize=9, pad=3)

            plot_joint_panel(
                ax_joint, ax_top, ax_right,
                rec, y_data,
                xlabel="Recurrence time",
                ylabel=y_label,
                threshold=thresh,
                cmap="plasma" if col_idx == 0 else "viridis",
            )

            # Record stats
            if len(rec) >= 10:
                r_p, _ = pearsonr(rec,  y_data)
                r_s, _ = spearmanr(rec, y_data)
                stats_rows.append({
                    "surface_point":    pt_idx + 1,
                    "coordinates":      coord_str,
                    "threshold":        thresh,
                    "quantity":         "peak" if col_idx == 0 else "mean",
                    "n_excursions":     len(rec),
                    "rec_time_mean":    rec.mean(),
                    "rec_time_std":     rec.std(),
                    "stress_mag_mean":  y_data.mean(),
                    "stress_mag_std":   y_data.std(),
                    "pearson_r":        r_p,
                    "spearman_r":       r_s,
                })

    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f"joint_pdf_point{pt_idx+1}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")

# ===========================================================================
# SAVE STATS CSV
# ===========================================================================

df_stats = pd.DataFrame(stats_rows, columns=[
    "surface_point", "coordinates", "threshold", "quantity",
    "n_excursions",
    "rec_time_mean", "rec_time_std",
    "stress_mag_mean", "stress_mag_std",
    "pearson_r", "spearman_r",
])
df_stats.to_csv(os.path.join(OUTPUT_DIR, "joint_stats.csv"), index=False)
print(f"\nSaved joint_stats.csv")

# Print correlation summary
print("\n" + "="*60)
print("Correlation summary  (Pearson r, Spearman r)")
print("="*60)
print(f"{'Point':<8} {'Threshold':<12} {'Quantity':<8} "
      f"{'Pearson':>10} {'Spearman':>10} {'N':>6}")
for _, row in df_stats.iterrows():
    print(f"  {int(row.surface_point):<6} {row.threshold:<12.4g} "
          f"{row.quantity:<8} "
          f"{row.pearson_r:>10.4f} {row.spearman_r:>10.4f} "
          f"{int(row.n_excursions):>6}")
print("="*60)