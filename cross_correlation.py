"""
step10_cross_correlation.py

Tests whether orientation at time t is predictive of traction at time t+s
for some lag s > 0.  A nonzero peak at positive lag means orientation
PRECEDES traction events — a physically meaningful result for biofilm
detachment since it implies you could anticipate high-stress events from
the particle's rotational state.

Drop this file into your project directory alongside correlation.py and
run it after generating grad_u.csv (and optionally grad_u_LW.csv).

Outputs (saved to ra_coupling_plots/):
  step10_crosscorr_overview.png      — 4-panel overview
  step10_crosscorr_table.png         — summary table of peak lags
  step10_crosscorr_conditional.png   — mean future traction given current orientation bin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
import os, sys
from fast_traction import build_transfer_matrix, fast_traction_magnitude

# ── CONFIG — should match your correlation.py ──────────────────
CSV_PATH    = "grad_u.csv"
LW_CSV_PATH = None      # set to None to skip LW
AXES        = [2.0, 1.0, 1.0]
MU          = 1.0
BURN_IN     = 0.10
OUTDIR      = "ra_coupling_plots/"
JEFFERY_PATH = "."

# Maximum cross-correlation lag to compute (in tau_eta units)
MAX_LAG_TIME = 5.0

# Number of orientation bins for the conditional mean traction plot
N_ORIENT_BINS = 5

# ── IMPORTS ────────────────────────────────────────────────────
sys.path.insert(0, JEFFERY_PATH)
from jeffery4    import Ellipsoid
from orientation import integrate_orientation

os.makedirs(OUTDIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# HELPERS (self-contained so this file runs standalone)
# ─────────────────────────────────────────────────────────────

def load_csv(path):
    df       = pd.read_csv(path)
    times    = df['time'].values
    A_series = np.zeros((len(df), 3, 3))
    for i in range(3):
        for j in range(3):
            A_series[:, i, j] = df[f'A{i+1}{j+1}'].values
    print(f"  Loaded {len(times)} timesteps  ({times[0]:.4f} → {times[-1]:.4f})")
    return times, A_series


def burn(arr, frac):
    n = int(len(arr) * frac)
    return arr[n:]


def symmetry_axis_angles(R_history):
    """
    Returns (theta_p, phi_p) in degrees from R_history (T, 3, 3).
    theta_p : polar angle of symmetry axis (0–180 deg)
    phi_p   : azimuthal angle (−180–180 deg)
    """
    p       = R_history[:, :, 0]
    theta_p = np.degrees(np.arccos(np.clip(p[:, 2], -1, 1)))
    phi_p   = np.degrees(np.arctan2(p[:, 1], p[:, 0]))
    return theta_p, phi_p

def cross_correlation(x, y, max_lag):
    """
    Normalised cross-correlation C_xy(lag) = <x'(t) y'(t+lag)> / (sigma_x sigma_y)
    for lag = 0, 1, ..., max_lag  (positive lag: y follows x).

    Uses FFT for efficiency.  Returns (lags_int, C_array).
    """
    x = x - np.mean(x)
    y = y - np.mean(y)
    sx = np.std(x); sy = np.std(y)
    if sx < 1e-12 or sy < 1e-12:
        return np.arange(max_lag + 1), np.zeros(max_lag + 1)

    n    = len(x)
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2

    Fx = np.fft.rfft(x, n=nfft)
    Fy = np.fft.rfft(y, n=nfft)

    # Cross-power spectrum: X* · Y  → inverse FFT gives cross-correlation
    # Positive lag means y lags x (i.e., x at time t, y at time t+lag)
    xcorr_full = np.fft.irfft(np.conj(Fx) * Fy)[:n]

    # Normalise by number of pairs at each lag and by sigma_x * sigma_y
    norm_counts = np.arange(n, 0, -1)
    xcorr_full /= (norm_counts * sx * sy)

    return np.arange(max_lag + 1), xcorr_full[:max_lag + 1]


def peak_lag(lags_dt, C):
    """
    Return the lag (in physical time) at which |C| is maximised.
    Also returns the peak value and its sign.
    """
    idx  = np.argmax(np.abs(C))
    return float(lags_dt[idx]), float(C[idx])


# ─────────────────────────────────────────────────────────────
# CORE ANALYSIS
# ─────────────────────────────────────────────────────────────

def compute_cross_correlations(times, A_series, R_history, tau_mag,
                                max_lag_time, label='CM'):
    """
    Compute normalised cross-correlations between orientation signals and traction.

    Pairs computed:
      cos(theta_p) → ||tau||   : polar angle of symmetry axis vs traction
      phi_p        → ||tau||   : azimuthal angle vs traction
      ||S||        → ||tau||   : strain rate vs traction (control/sanity check)
      ||W||        → ||tau||   : vorticity rate vs traction (control)

    Returns dict: signal_name → {'lags_dt', 'C', 'peak_lag', 'peak_val'}
    """
    dt      = float(times[1] - times[0])
    max_lag = min(int(max_lag_time / dt), len(times) - 1)
    lags_dt = np.arange(max_lag + 1) * dt

    theta_p, phi_p = symmetry_axis_angles(R_history)

    S = 0.5 * (A_series + A_series.transpose(0, 2, 1))
    W = 0.5 * (A_series - A_series.transpose(0, 2, 1))
    S_norm = np.sqrt(np.einsum('tij,tij->t', S, S))
    W_norm = np.sqrt(np.einsum('tij,tij->t', W, W))

    pairs = {
        '$\\cos\\theta_p$':    np.cos(np.radians(theta_p)),
        '$\\phi_p$':           phi_p,
        '$\\|S\\|$ (control)': S_norm,
        '$\\|W\\|$ (control)': W_norm,
    }

    results = {}
    print(f"\n  [{label}] Cross-correlation peak lags (x → ||tau||, positive lag = x precedes tau):")

    for name, x_sig in pairs.items():
        _, C = cross_correlation(x_sig, tau_mag, max_lag)
        pl, pv = peak_lag(lags_dt, C)
        results[name] = {
            'lags_dt':  lags_dt,
            'C':        C,
            'peak_lag': pl,
            'peak_val': pv,
        }
        print(f"    {name:<28}  peak at lag={pl:.4f} tau_eta  "
              f"C_peak={pv:.4f}")

    return results


def conditional_mean_future_traction(times, theta_p, tau_mag,
                                     lags_tau_eta, n_bins=5):
    """
    For each orientation bin of theta_p at time t, compute the mean traction
    at time t + lag for a set of lags.

    This answers: "if the particle is in orientation bin k right now,
    what is the expected traction in the near future?"

    Returns:
      bin_centres : (n_bins,) array of theta_p bin centres in degrees
      mean_tau    : (n_bins, n_lags) array of mean future traction
      lags        : (n_lags,) array of lag values used
    """
    dt     = float(times[1] - times[0])
    n      = len(tau_mag)

    bin_edges   = np.linspace(theta_p.min(), theta_p.max(), n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    lag_steps = [int(round(lag / dt)) for lag in lags_tau_eta]
    mean_tau  = np.full((n_bins, len(lag_steps)), np.nan)

    for k_lag, k_step in enumerate(lag_steps):
        if k_step >= n:
            continue
        # For each t in 0..n-k_step-1, orientation at t and traction at t+k_step
        t_indices = np.arange(n - k_step)
        th_now    = theta_p[t_indices]
        tau_future = tau_mag[t_indices + k_step]

        for b in range(n_bins):
            mask = (th_now >= bin_edges[b]) & (th_now < bin_edges[b + 1])
            if mask.sum() > 10:
                mean_tau[b, k_lag] = np.mean(tau_future[mask])

    return bin_centres, mean_tau, np.array(lags_tau_eta)


# ─────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────

def plot_crosscorr_overview(xcorr_cm, xcorr_lw=None, tau_K=None):
    """
    4-panel plot showing the cross-correlation functions for each pair.
    CM in orange (solid), LW in blue (dashed) if provided.
    """
    cm_color = '#E07B39'
    lw_color = '#4472C4'
    signals  = list(xcorr_cm.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = list(axes.flat)

    for ax, name in zip(axes_flat, signals):
        d_cm = xcorr_cm[name]
        lags = d_cm['lags_dt']
        C_cm = d_cm['C']

        ax.plot(lags, C_cm, color=cm_color, lw=2.2,
                label=f"CM  peak lag={d_cm['peak_lag']:.3f}, "
                      f"C={d_cm['peak_val']:.3f}")

        if xcorr_lw is not None and name in xcorr_lw:
            d_lw = xcorr_lw[name]
            ax.plot(d_lw['lags_dt'], d_lw['C'],
                    color=lw_color, lw=2.2, linestyle='--',
                    label=f"LW  peak lag={d_lw['peak_lag']:.3f}, "
                          f"C={d_lw['peak_val']:.3f}")

        ax.axhline(0, color='grey', lw=0.8)
        ax.axvline(0, color='grey', lw=0.8, linestyle=':')

        # Mark peak lag for CM
        pl = d_cm['peak_lag']
        pv = d_cm['peak_val']
        ax.axvline(pl, color=cm_color, lw=1.2, linestyle='--', alpha=0.6)
        ax.plot(pl, pv, 'o', color=cm_color, ms=8, zorder=5)

        if xcorr_lw is not None and name in xcorr_lw:
            d_lw = xcorr_lw[name]
            pl_lw = d_lw['peak_lag']
            pv_lw = d_lw['peak_val']
            ax.axvline(pl_lw, color=lw_color, lw=1.2, linestyle='--', alpha=0.6)
            ax.plot(pl_lw, pv_lw, 's', color=lw_color, ms=8, zorder=5)

        if tau_K is not None:
            ax.axvline(tau_K, color='black', lw=1.0, linestyle='-.',
                       alpha=0.5, label=f'$\\tau_K$={tau_K:.3f}')

        ax.set_xlim(0, lags[-1])
        ax.set_xlabel('Lag $s$  ($\\tau_\\eta$)', fontsize=11)
        ax.set_ylabel('$C_{xy}(s)$', fontsize=11)
        ax.set_title(f'{name}  →  $\\|\\tau\\|$', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Main interpretation annotation on fig
    fig.suptitle(
        'Cross-correlation: orientation / flow scalar  →  traction  $\\|\\tau\\|$\n'
        'Positive peak lag $s > 0$ means the signal at time $t$ '
        'predicts traction at $t+s$ — orientation precedes traction events',
        fontsize=12, fontweight='bold'
    )
    fig.tight_layout()
    fig.savefig(OUTDIR + 'step10_crosscorr_overview.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved step10_crosscorr_overview.png")


def plot_crosscorr_table(xcorr_cm, xcorr_lw=None, tau_K=None):
    """Summary table of peak lags and peak values."""
    signals   = list(xcorr_cm.keys())
    col_labels = ['Signal  (x → ||tau||)',
                  'CM  peak lag ($\\tau_\\eta$)',
                  'CM  C_peak',
                  'LW  peak lag ($\\tau_\\eta$)',
                  'LW  C_peak']

    rows = []
    for name in signals:
        d_cm = xcorr_cm[name]
        row  = [name.replace('$', '').replace('\\|', '|')
                    .replace('\\cos', 'cos').replace('\\theta_p', 'theta_p'),
                f"{d_cm['peak_lag']:.4f}",
                f"{d_cm['peak_val']:.4f}"]
        if xcorr_lw is not None and name in xcorr_lw:
            d_lw = xcorr_lw[name]
            row += [f"{d_lw['peak_lag']:.4f}", f"{d_lw['peak_val']:.4f}"]
        else:
            row += ['--', '--']
        rows.append(row)

    fig, ax = plt.subplots(figsize=(13, 3 + 0.6 * len(rows)))
    ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   cellLoc='center', loc='center',
                   colColours=['#DDEEFF'] * len(col_labels))
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 2.0)

    note = (
        "Peak lag > 0: orientation at time t predicts traction at t + lag  "
        "(causal precursor)\n"
        "Peak lag ≈ 0: contemporaneous correlation only  "
        "(no predictive lead)\n"
        "C_peak large: strong predictive relationship"
    )
    if tau_K is not None:
        note += f"\nKolmogorov time tau_K = {tau_K:.4f}"

    ax.set_title(
        'Cross-correlation summary: peak lags and amplitudes\n' + note,
        fontsize=10, fontweight='bold', pad=20
    )
    fig.tight_layout()
    fig.savefig(OUTDIR + 'step10_crosscorr_table.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved step10_crosscorr_table.png")


def plot_conditional_mean_future_traction(bin_centres, mean_tau, lags,
                                          label='CM'):
    """
    Heatmap and line plot showing mean future traction E[||tau||(t+s) | theta_p(t) in bin].

    Rows = orientation bins, columns = lags.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # ── Left: heatmap ──────────────────────────────────────────
    ax  = axes[0]
    im  = ax.imshow(mean_tau, aspect='auto', cmap='inferno',
                    origin='lower',
                    extent=[lags[0], lags[-1],
                            bin_centres[0], bin_centres[-1]])
    plt.colorbar(im, ax=ax, label='$E[\\|\\tau\\|(t+s)]$')
    ax.set_xlabel('Future lag $s$  ($\\tau_\\eta$)', fontsize=11)
    ax.set_ylabel('$\\theta_p(t)$ bin centre (deg)', fontsize=11)
    ax.set_title(
        f'[{label}] Mean future traction given current orientation\n'
        '$E[\\|\\tau\\|(t+s) \\mid \\theta_p(t)]$',
        fontsize=11, fontweight='bold'
    )
    ax.grid(False)

    # ── Right: line plot per orientation bin ───────────────────
    ax     = axes[1]
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(bin_centres)))
    for b, (bc, col) in enumerate(zip(bin_centres, colors)):
        row = mean_tau[b]
        mask = np.isfinite(row)
        if mask.sum() < 2:
            continue
        ax.plot(lags[mask], row[mask], color=col, lw=2.0,
                label=f'$\\theta_p \\approx {bc:.0f}°$')
    ax.set_xlabel('Future lag $s$  ($\\tau_\\eta$)', fontsize=11)
    ax.set_ylabel('$E[\\|\\tau\\|(t+s)]$', fontsize=11)
    ax.set_title(
        f'[{label}] Mean future traction per orientation bin\n'
        'Separation between lines = orientation effect on future stress',
        fontsize=11, fontweight='bold'
    )
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        'Conditional mean future traction  '
        '$E[\\|\\tau\\|(t+s) \\mid \\theta_p(t)]$\n'
        'If lines separate and diverge with lag: '
        'orientation is a leading indicator of traction',
        fontsize=12, fontweight='bold'
    )
    fig.tight_layout()
    fname = OUTDIR + f'step10_crosscorr_conditional_{label.lower()}.png'
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {fname}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':

    print("── Step 10: Cross-correlation  orientation → traction ──")

    # ── Load and prep CM data ───────────────────────────────────
    print("\nLoading CM data...")
    times_cm, A_cm = load_csv(CSV_PATH)
    n_burn          = int(len(times_cm) * BURN_IN)

    print("Integrating orientation ODE (CM)...")
    R_cm, omega_cm, *_ = integrate_orientation(AXES, A_cm, times_cm)

    times_b = times_cm[n_burn:]
    A_b     = A_cm[n_burn:]
    R_b     = R_cm[n_burn:]

    x_tip = np.array([AXES[0], 0., 0.])
    print("Building transfer matrix (CM, done once)...")
    M_cm, n_hat_cm = build_transfer_matrix(AXES, x_tip, mu=MU)
    print("Computing traction magnitude (CM, fast)...")
    tau_cm = fast_traction_magnitude(M_cm, n_hat_cm, A_b, R_b)

    # ── Kolmogorov time ─────────────────────────────────────────
    AtA   = np.einsum('tji,tjk->tik', A_b, A_b)
    tau_K = 1.0 / np.sqrt(np.mean(np.einsum('tii->t', AtA)))
    print(f"  tau_K = {tau_K:.4f}")

    # ── Cross-correlations (CM) ─────────────────────────────────
    xcorr_cm = compute_cross_correlations(
        times_b, A_b, R_b, tau_cm,
        max_lag_time=MAX_LAG_TIME, label='CM'
    )

    # ── LW data (optional) ─────────────────────────────────────
    xcorr_lw = None
    tau_lw   = None
    R_lw_b   = None
    times_lw_b = None

    if LW_CSV_PATH is not None and os.path.exists(LW_CSV_PATH):
        print(f"\nLoading LW data from '{LW_CSV_PATH}'...")
        times_lw, A_lw = load_csv(LW_CSV_PATH)
        n_burn_lw       = int(len(times_lw) * BURN_IN)

        print("Integrating orientation ODE (LW)...")
        R_lw, *_ = integrate_orientation(AXES, A_lw, times_lw)

        times_lw_b = times_lw[n_burn_lw:]
        A_lw_b     = A_lw[n_burn_lw:]
        R_lw_b     = R_lw[n_burn_lw:]

        print("Building transfer matrix (LW, done once)...")
        M_lw, n_hat_lw = build_transfer_matrix(AXES, x_tip, mu=MU)
        print("Computing traction magnitude (LW, fast)...")
        tau_lw = fast_traction_magnitude(M_lw, n_hat_lw, A_lw_b, R_lw_b)

        xcorr_lw = compute_cross_correlations(
            times_lw_b, A_lw_b, R_lw_b, tau_lw,
            max_lag_time=MAX_LAG_TIME, label='LW'
        )
    elif LW_CSV_PATH is not None:
        print(f"\n  '{LW_CSV_PATH}' not found — plotting CM only.")

    # ── Plots ───────────────────────────────────────────────────
    print("\nGenerating plots...")

    plot_crosscorr_overview(xcorr_cm, xcorr_lw, tau_K=tau_K)
    plot_crosscorr_table(xcorr_cm, xcorr_lw, tau_K=tau_K)

    # Conditional mean future traction — evaluated at several lags
    theta_p_cm, _ = symmetry_axis_angles(R_b)
    lags_to_check  = np.linspace(0.0, MAX_LAG_TIME, 20)

    bin_c, mean_tau_mat, lags_used = conditional_mean_future_traction(
        times_b, theta_p_cm, tau_cm, lags_to_check,
        n_bins=N_ORIENT_BINS
    )
    plot_conditional_mean_future_traction(bin_c, mean_tau_mat, lags_used,
                                          label='CM')

    if xcorr_lw is not None:
        theta_p_lw, _ = symmetry_axis_angles(R_lw_b)
        bin_c_lw, mean_tau_mat_lw, _ = conditional_mean_future_traction(
            times_lw_b, theta_p_lw, tau_lw, lags_to_check,
            n_bins=N_ORIENT_BINS
        )
        plot_conditional_mean_future_traction(bin_c_lw, mean_tau_mat_lw,
                                              lags_used, label='LW')

    print(f"\nAll step 10 plots saved to '{OUTDIR}'")
    print("Done.")