"""
mfpt_aspect_ratio.py
====================
Computes MFPT vs threshold for varying ellipsoid aspect ratio.

For each aspect ratio a1 in ASPECT_RATIOS, the ellipsoid has semi-axes
[a1, 1, 1] — i.e. the two short axes are fixed at 1 and the long axis varies.

For each aspect ratio:
  1. Build the stress transfer matrix at (a1, 0, 0) — the tip of the long axis
  2. Compute ||sigma||_F time series
  3. Compute MFPT at each threshold (auto-selected as quantiles)
  4. Fit a line to log(MFPT) vs threshold to characterise the growth rate

All MFPT vs threshold curves are overlaid on a single figure, coloured by
aspect ratio, with both linear-y and log-y panels.

The fitted line for each aspect ratio is:
    log(MFPT) = slope * threshold + intercept
i.e.  MFPT ~ exp(slope * threshold)
which characterises how quickly rare events become rarer as threshold increases.

Outputs
-------
  aspect_ratio/mfpt_vs_ar.png          -- all curves overlaid (linear + log y)
  aspect_ratio/mfpt_lines.png          -- slope and intercept vs aspect ratio
  aspect_ratio/mfpt_aspect_ratio.csv   -- MFPT values per aspect ratio + threshold
  aspect_ratio/mfpt_line_fits.csv      -- fitted line parameters per aspect ratio
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import os
import time
import warnings
from scipy.stats import linregress

from jeffery4_2  import Ellipsoid
from orientation_2 import integrate_orientation

warnings.filterwarnings("ignore")

# ===========================================================================
# CONFIGURATION
# ===========================================================================

CSV_PATH   = "grad_u.csv"
OUTPUT_DIR = "aspect_ratio"
DT         = 0.001   # physical timestep

# Long-axis values to sweep — short axes fixed at 1
ASPECT_RATIOS = [2, 3, 4, 5, 6, 7, 8, 9, 10]

# Quantile levels for threshold selection — same for all aspect ratios
# so that comparisons are on equal footing
QUANTILE_LEVELS = np.linspace(0.50, 0.95, 8)

# Minimum number of recurrence events to include a threshold
MIN_EVENTS = 20

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================================================================
# LOAD DATA ONCE
# ===========================================================================

print("Loading grad_u.csv...")
df = pd.read_csv(CSV_PATH)
t  = df["time"].values * DT
T  = len(t)

A_series = np.zeros((T, 3, 3))
for i in range(3):
    for j in range(3):
        A_series[:, i, j] = df[f"A{i+1}{j+1}"].values

print(f"  {T} timesteps,  t in [{t[0]:.4g}, {t[-1]:.4g}] (physical time)")

# ===========================================================================
# UTILITIES
# ===========================================================================

def build_stress_transfer_matrix(ellipsoid_class, a, mu, x_surface):
    """9x9 transfer matrix:  vec(sigma) = M @ vec(A_body)."""
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


def recurrence_times(frob, t, threshold):
    """Waiting times between successive upcrossings of frob = threshold."""
    above     = frob > threshold
    crossings = np.where(~above[:-1] & above[1:])[0] + 1
    if len(crossings) < 2:
        return np.array([])
    return np.diff(t[crossings])


def rotate_A_body(A_series, R_history):
    """Rotate A into body frame: A_body = R^T A R."""
    tmp   = np.einsum('tji,tjk->tik', R_history, A_series)
    return np.einsum('tij,tjk->tik', tmp, R_history)

# ===========================================================================
# MAIN LOOP OVER ASPECT RATIOS
# ===========================================================================

mfpt_rows = []
line_rows = []

# Colourmap: one colour per aspect ratio
cmap    = cm.viridis
colours = [cmap(v) for v in np.linspace(0.05, 0.95, len(ASPECT_RATIOS))]

for ar_idx, a1 in enumerate(ASPECT_RATIOS):
    a   = np.array([float(a1), 1.0, 1.0])
    mu  = 1.0
    tip = np.array([float(a1), 0.0, 0.0])   # (a1, 0, 0)

    print(f"\n{'='*50}")
    print(f"Aspect ratio a1 = {a1}  (axes = [{a1}, 1, 1])")
    print(f"{'='*50}")

    # -- Integrate orientation ODE for this ellipsoid geometry ---------------
    print("  Integrating orientation ODE...", end=" ", flush=True)
    t0 = time.perf_counter()
    R_history, *_ = integrate_orientation(a, A_series, t)
    print(f"done in {time.perf_counter()-t0:.2f}s")

    # -- Rotate A into body frame --------------------------------------------
    A_body_ts  = rotate_A_body(A_series, R_history)
    vec_A_body = A_body_ts.reshape(T, 9)

    # -- Build transfer matrix at tip (a1, 0, 0) -----------------------------
    print(f"  Building transfer matrix at ({a1}, 0, 0)...", end=" ", flush=True)
    t0 = time.perf_counter()
    M  = build_stress_transfer_matrix(Ellipsoid, a, mu, tip)
    print(f"done in {time.perf_counter()-t0:.2f}s")

    # -- Frobenius norm time series ------------------------------------------
    frob = np.linalg.norm(vec_A_body @ M.T, axis=1)
    print(f"  ||sigma||_F:  min={frob.min():.3f}  "
          f"mean={frob.mean():.3f}  max={frob.max():.3f}")

    # -- Select thresholds as quantiles of this frob distribution ------------
    thresholds = np.quantile(frob, QUANTILE_LEVELS)

    # -- Compute MFPT at each threshold --------------------------------------
    mfpt_vals  = []
    thresh_vals = []

    for thresh in thresholds:
        wts = recurrence_times(frob, t, thresh)
        if len(wts) < MIN_EVENTS:
            continue
        mfpt = wts.mean()
        mfpt_vals.append(mfpt)
        thresh_vals.append(thresh)
        mfpt_rows.append({
            "a1":            a1,
            "axes":          f"[{a1}, 1, 1]",
            "threshold":     thresh,
            "n_events":      len(wts),
            "mfpt":          mfpt,
        })
        print(f"    c={thresh:.3f}:  {len(wts)} events  "
              f"MFPT={mfpt:.5f}")

    if len(thresh_vals) < 2:
        print(f"  Not enough thresholds with >= {MIN_EVENTS} events — skipping fit")
        line_rows.append({
            "a1": a1, "axes": f"[{a1}, 1, 1]",
            "slope": np.nan, "intercept": np.nan, "r2": np.nan,
        })
        continue

    # -- Fit line to log(MFPT) vs threshold ----------------------------------
    log_mfpt = np.log(mfpt_vals)
    slope, intercept, r, _, _ = linregress(thresh_vals, log_mfpt)
    r2 = r**2

    print(f"  Line fit:  log(MFPT) = {slope:.4f} * c + {intercept:.4f}  "
          f"(R² = {r2:.4f})")

    line_rows.append({
        "a1":        a1,
        "axes":      f"[{a1}, 1, 1]",
        "slope":     slope,
        "intercept": intercept,
        "r2":        r2,
    })

# ===========================================================================
# PLOT 1 — MFPT vs threshold, all aspect ratios overlaid
# ===========================================================================

df_mfpt  = pd.DataFrame(mfpt_rows)
df_lines = pd.DataFrame(line_rows)

fig, axes = plt.subplots(1, 2, figsize=(14, 10))
fig.suptitle(
    "MFPT vs threshold — varying aspect ratio\n"
    "Surface point at (a1, 0, 0)  |  axes = [a1, 1, 1]",
    fontsize=13,
)

for ar_idx, a1 in enumerate(ASPECT_RATIOS):
    sub    = df_mfpt[df_mfpt["a1"] == a1].sort_values("threshold")
    if sub.empty:
        continue
    colour = colours[ar_idx]
    label  = f"a1={a1}"

    axes[0].plot(sub["threshold"], sub["mfpt"],
                 "o-", color=colour, lw=1.8, ms=5, label=label)
    # Right panel: log x, linear y
    axes[1].plot(sub["threshold"], sub["mfpt"],
                 "o-", color=colour, lw=1.8, ms=5, label=label)

    # Overlay the fitted line on the log-x panel
    lr = df_lines[df_lines["a1"] == a1].iloc[0]
    if not np.isnan(lr["slope"]):
        c_fit    = np.geomspace(sub["threshold"].min(),
                                sub["threshold"].max(), 200)
        mfpt_fit = np.exp(lr["slope"] * c_fit + lr["intercept"])
        axes[1].plot(c_fit, mfpt_fit, "--", color=colour, lw=1.0, alpha=0.6)

axes[0].set_xlabel("threshold  c  (||sigma||_F)")
axes[0].set_ylabel("MFPT  (physical time)")
axes[0].set_title("Linear x, linear y")
axes[0].legend(fontsize=8, ncol=2)

axes[1].set_xscale("log")
axes[1].set_xlabel("threshold  c  (||sigma||_F, log scale)")
axes[1].set_ylabel("MFPT  (physical time)")
axes[1].set_title("Log x, linear y  —  dashed = fitted exp line")
axes[1].legend(fontsize=8, ncol=2)

plt.tight_layout()

# Colourbar underneath both panels
fig.subplots_adjust(bottom=0.18)
cax = fig.add_axes([0.25, 0.04, 0.50, 0.03])   # [left, bottom, width, height]
sm  = plt.cm.ScalarMappable(
    cmap=cmap,
    norm=plt.Normalize(vmin=ASPECT_RATIOS[0], vmax=ASPECT_RATIOS[-1])
)
sm.set_array([])
fig.colorbar(sm, cax=cax, orientation="horizontal",
             label="aspect ratio  a1")
 
fname = os.path.join(OUTPUT_DIR, "mfpt_vs_ar.png")
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {fname}")

# ===========================================================================
# PLOT 2 — Fitted line parameters vs aspect ratio + equation estimates
# ===========================================================================
# For each of slope(a1) and intercept(a1) we try three functional forms:
#   linear:      f(a) = p0 + p1*a
#   power law:   f(a) = p0 * a^p1       (fit in log-log)
#   exponential: f(a) = p0 * exp(p1*a)  (fit in log-linear)
# The best fit is selected by R^2 on the original (not log) scale.
# ===========================================================================

from scipy.stats import linregress as _linregress
 
df_lines_valid = df_lines.dropna(subset=["slope"])
a1_vals = df_lines_valid["a1"].values.astype(float)
 
 
def fit_power(x, y):
    """Fit y = p0 * x^p1 via log-log linear regression. Returns p0, p1, R²."""
    sign  = 1 if np.all(y > 0) else -1
    log_y = np.log(np.abs(y))
    log_x = np.log(x)
    p1, log_p0, r, _, _ = _linregress(log_x, log_y)
    p0    = sign * np.exp(log_p0)
    y_pred = p0 * x**p1
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return p0, p1, r2
 
def fit_log(x, y):
    """
    Fit y = p0 + p1*ln(a) via two methods and return the better one.
 
    Method 1: linear regression on log(x)  — exact, closed form
              linregress returns (slope, intercept) = (p1, p0)
    Method 2: scipy curve_fit              — nonlinear least squares,
              better numerical conditioning for small datasets
 
    R² is computed on the original (untransformed) scale in both cases.
    """
    from scipy.optimize import curve_fit as _curve_fit
 
    log_x = np.log(x)
    ss_tot = np.sum((y - y.mean())**2)
 
    results = []
 
    # Method 1: exact linear regression on ln(x)
    try:
        p1_lr, p0_lr, *_ = _linregress(log_x, y)   # slope=p1, intercept=p0
        y_pred = p0_lr + p1_lr * log_x
        ss_res = np.sum((y - y_pred)**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        results.append((p0_lr, p1_lr, r2))
    except Exception:
        pass
 
    # Method 2: curve_fit with explicit function
    try:
        def log_fn(a, p0, p1):
            return p0 + p1 * np.log(a)
        p0_init = y.mean()
        p1_init = (y[-1] - y[0]) / (log_x[-1] - log_x[0] + 1e-12)
        popt, _ = _curve_fit(log_fn, x, y, p0=[p0_init, p1_init], maxfev=10000)
        y_pred  = log_fn(x, *popt)
        ss_res  = np.sum((y - y_pred)**2)
        r2      = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        results.append((popt[0], popt[1], r2))
    except Exception:
        pass
 
    if not results:
        return 0.0, 0.0, 0.0
 
    # Return whichever method gave higher R²
    return max(results, key=lambda t: t[2])
 
def plot_slope_panel(ax, x, y):
    """Power law fit for slope(a1)."""
    ax.plot(x, y, "o", color="steelblue", ms=7, zorder=5)
    p0, p1, r2 = fit_power(x, y)
    x_dense = np.linspace(x.min(), x.max(), 300)
    ax.plot(x_dense, p0 * x_dense**p1,
            "-", color="steelblue", lw=2.2,
            label=f"{p0:.4f} × a¹ ⁿ  (n={p1:.4f},  R²={r2:.3f})")
    ax.set_xlabel("aspect ratio  a1")
    ax.set_ylabel("slope")
    ax.set_title("Slope vs a1 fit:  slope = p₀ × aⁿ")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    label = f"power:  {p0:.4f} * a^{p1:.4f}  (R²={r2:.3f})"
    print(f"  slope(a1)     = {label}")
    return label
 
 
def plot_intercept_panel(ax, x, y):
    """Log fit for intercept(a1)."""
    ax.plot(x, y, "o", color="tomato", ms=7, zorder=5)
    p0, p1, r2 = fit_log(x, y)
    x_dense = np.linspace(x.min(), x.max(), 300)
    ax.plot(x_dense, p0 + p1 * np.log(x_dense),
            "-", color="tomato", lw=2.2,
            label=f"{p0:.4f} + {p1:.4f}×ln(a)  (R²={r2:.3f})")
    ax.set_xlabel("aspect ratio  a1")
    ax.set_ylabel("intercept")
    ax.set_title("Intercept vs a1 fit:  intercept = p₀ + p₁×ln(a)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    label = f"log:  {p0:.4f} + {p1:.4f}*ln(a)  (R²={r2:.3f})"
    print(f"  intercept(a1) = {label}")
    return label
 
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle(
    "Fitted line parameters  log(MFPT) = slope * c + intercept\n"
    "vs aspect ratio  a1  (with equation estimates)",
    fontsize=13,
)
 
print("\nBest-fit equations for slope and intercept:")
label_slope      = plot_slope_panel(axes[0], a1_vals,
                                    df_lines_valid["slope"].values)
label_intercept  = plot_intercept_panel(axes[1], a1_vals,
                                        df_lines_valid["intercept"].values)
 
axes[2].plot(df_lines_valid["a1"], df_lines_valid["r2"],
             "o-", color="seagreen", lw=2, ms=7)
axes[2].set_xlabel("aspect ratio  a1")
axes[2].set_ylabel("R²")
axes[2].set_title("R² of log-linear fit\n"
                  "(1 = perfectly exponential MFPT growth)")
axes[2].set_ylim(0, 1.05)
axes[2].grid(True, alpha=0.3)
 
plt.tight_layout()
fname = os.path.join(OUTPUT_DIR, "mfpt_lines.png")
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {fname}")
 
print(f"  slope(a1)     = {label_slope}")
print(f"  intercept(a1) = {label_intercept}")

# ===========================================================================
# SAVE CSVs
# ===========================================================================

df_mfpt.to_csv(os.path.join(OUTPUT_DIR, "mfpt_aspect_ratio.csv"), index=False)
df_lines.to_csv(os.path.join(OUTPUT_DIR, "mfpt_line_fits.csv"), index=False)

print(f"Saved mfpt_aspect_ratio.csv")
print(f"Saved mfpt_line_fits.csv")

# ===========================================================================
# SUMMARY TABLE
# ===========================================================================

print("\n" + "="*60)
print("Line fit summary:  log(MFPT) = slope * c + intercept")
print("="*60)
print(f"  {'a1':>4}  {'slope':>10}  {'intercept':>12}  {'R²':>8}")
for _, row in df_lines.iterrows():
    if np.isnan(row["slope"]):
        print(f"  {int(row['a1']):>4}  {'---':>10}  {'---':>12}  {'---':>8}")
    else:
        print(f"  {int(row['a1']):>4}  {row['slope']:>10.4f}  "
              f"{row['intercept']:>12.4f}  {row['r2']:>8.4f}")
print("="*60)