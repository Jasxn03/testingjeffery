"""
step12_intermittency_QR.py

Two analyses:

1. TRACTION INTERMITTENCY vs ASPECT RATIO
   Compute the flatness (normalised 4th moment) and kurtosis of the
   traction distribution at the tip (a,0,0) as a function of aspect
   ratio r = a/b.  High flatness/kurtosis means rare but extreme
   traction events dominate — directly relevant to detachment.

   Flatness  F = <tau^4> / <tau^2>^2
   Kurtosis  K = F - 3  (excess kurtosis; K=0 for Gaussian)

   Also plots the full traction PDF for each aspect ratio.

2. TRACTION CONDITIONED ON Q-R INVARIANT SPACE
   Colour the Q-R plane by mean traction, showing which regions of
   velocity gradient space produce the most dangerous surface stress.
   Q = 0.5*(||W||^2 - ||S||^2)  (Q>0: rotation dominated)
   R = -1/3 Tr(A^3)              (R>0: strain stretching)

   Classic teardrop shape expected; colouring by traction is new.

Both analyses use fast_traction.py for speed.
LW model comparison enabled by setting LW_CSV_PATH to a file path.

Outputs saved to ra_coupling_plots/:
  step12a_intermittency_pdfs.png
  step12a_flatness_kurtosis_vs_r.png
  step12b_QR_traction.png
  step12b_QR_traction_scatter.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
import os, sys

# ── CONFIG ─────────────────────────────────────────────────────
CSV_PATH    = "grad_u.csv"
LW_CSV_PATH = None          # set to e.g. "grad_u_LW.csv" to enable
AXES        = [2.0, 1.0, 1.0]
MU          = 1.0
BURN_IN     = 0.10
OUTDIR      = "ra_coupling_plots/"
JEFFERY_PATH = "."

# Aspect ratios to sweep for intermittency analysis
# tip point is always (r, 0, 0) — semi-axes [r, 1, 1]
ASPECT_RATIOS = [0.25, 0.5, 0.6, 0.75, 0.9, 1.0, 1.5, 2.0, 2.25, 2.5, 3.0, 4.0, 6.0, 8.0]

# Q-R binning
N_QR_BINS = 100      # number of bins along each axis of Q-R plane

# ── IMPORTS ────────────────────────────────────────────────────
sys.path.insert(0, JEFFERY_PATH)
from orientation   import integrate_orientation
from fast_traction import build_transfer_matrix, fast_traction_magnitude

os.makedirs(OUTDIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def load_csv(path):
    df       = pd.read_csv(path)
    times    = df['time'].values
    A_series = np.zeros((len(df), 3, 3))
    for i in range(3):
        for j in range(3):
            A_series[:, i, j] = df[f'A{i+1}{j+1}'].values
    print(f"  Loaded {len(times)} timesteps  ({times[0]:.4f} -> {times[-1]:.4f})")
    return times, A_series


def apply_burnin(times, A_series, R_history, frac):
    n = int(len(times) * frac)
    return times[n:], A_series[n:], R_history[n:]


def flow_scalars(A_series):
    """Compute Q, R invariants and ||S||, ||W|| from A time series."""
    S      = 0.5 * (A_series + A_series.transpose(0, 2, 1))
    W      = 0.5 * (A_series - A_series.transpose(0, 2, 1))
    S_norm = np.sqrt(np.einsum('tij,tij->t', S, S))
    W_norm = np.sqrt(np.einsum('tij,tij->t', W, W))
    Q      = 0.5 * (W_norm**2 - S_norm**2)
    A2     = np.einsum('tij,tjk->tik', A_series, A_series)
    A3     = np.einsum('tij,tjk->tik', A2, A_series)
    R_inv  = -np.einsum('tii->t', A3) / 3.0
    return {'S_norm': S_norm, 'W_norm': W_norm, 'Q': Q, 'R_inv': R_inv}


def flatness_kurtosis(x, n_bootstrap=200):
    mu  = np.mean(x)
    if mu < 1e-30:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    xn  = x / mu - 1.0
    m2  = np.mean(xn**2)
    m4  = np.mean(xn**4)
    if m2 < 1e-30:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    F = m4 / m2**2
    K = F - 3.0
    I = float(np.percentile(x, 95) / np.percentile(x, 50))

    # Bootstrap std of flatness to quantify uncertainty
    rng   = np.random.default_rng(42)
    F_boot = []
    for _ in range(n_bootstrap):
        idx  = rng.integers(0, len(xn), size=len(xn))
        xb   = xn[idx]
        m2b  = np.mean(xb**2)
        m4b  = np.mean(xb**4)
        if m2b > 1e-30:
            F_boot.append(m4b / m2b**2)
    F_std = float(np.std(F_boot)) if F_boot else np.nan

    return float(F), float(K), float(I), F_std


def kolmogorov_time(A_series):
    AtA  = np.einsum('tji,tjk->tik', A_series, A_series)
    return 1.0 / np.sqrt(np.mean(np.einsum('tii->t', AtA)))


# ─────────────────────────────────────────────────────────────
# ANALYSIS 1 — INTERMITTENCY vs ASPECT RATIO
# ─────────────────────────────────────────────────────────────

def compute_intermittency_vs_r(A_series, R_history, aspect_ratios, times, mu=1.0):
    """
    For each aspect ratio r, build ellipsoid [r,1,1], compute traction
    at the tip (r,0,0) using the transfer matrix, and compute
    flatness and kurtosis of the traction distribution.

    Returns dict: r -> {
        'tau_mag': (T,) array,
        'flatness': float,
        'kurtosis': float,
        'mean': float,
        'std': float,
        'skewness': float,
    }
    """
    results = {}
    for r in aspect_ratios:
        axes  = [float(r), 1.0, 1.0]
        x_tip = np.array([float(r), 0., 0.])
        print(f"  r={r:.2f}  building M...", end=" ", flush=True)
        R_r, *_ = integrate_orientation(axes, A_series, times)

        M, n_hat = build_transfer_matrix(axes, x_tip, mu=mu)
        tau      = fast_traction_magnitude(M, n_hat, A_series, R_r)

        F, K, I, F_std = flatness_kurtosis(tau)
        skew = float(np.mean((tau - np.mean(tau))**3) / np.std(tau)**3)

        results[r] = {
            'tau_mag':  tau,
            'flatness': F,
            'kurtosis': K,
            'intermittency': I,
            'flatness_std': F_std,
            'kurtosis_std': F_std, #K = F-3 so same
            'mean':     float(np.mean(tau)),
            'std':      float(np.std(tau)),
            'skewness': skew,
        }
        print(f"F={F:.3f}±{F_std:.3f}  K={K:.3f}  skew={skew:.3f}")

    return results


def plot_intermittency_pdfs(results_cm, results_lw=None):
    """
    PDF of traction for each aspect ratio — CM solid, LW dashed.
    One panel per aspect ratio, arranged in a grid.
    """
    rs      = list(results_cm.keys())
    n_r     = len(rs)
    ncols   = 5
    nrows   = int(np.ceil(n_r / ncols))
    cm_color = '#E07B39'
    lw_color = '#4472C4'

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4*ncols, 3.5*nrows))
    axes_flat = list(axes.flat) if nrows > 1 else list(axes)

    for ax, r in zip(axes_flat, rs):
        tau_cm = results_cm[r]['tau_mag']
        p1, p99 = np.percentile(tau_cm, 1), np.percentile(tau_cm, 99)
        tc = tau_cm[(tau_cm >= p1) & (tau_cm <= p99)]
        kde = gaussian_kde(tc, bw_method=0.1)
        xg  = np.linspace(p1, p99, 400)
        ax.plot(xg, kde(xg), color=cm_color, lw=2.0, label='CM')
        ax.fill_between(xg, kde(xg), alpha=0.15, color=cm_color)

        if results_lw is not None and r in results_lw:
            tau_lw = results_lw[r]['tau_mag']
            p1l, p99l = np.percentile(tau_lw, 1), np.percentile(tau_lw, 99)
            tl = tau_lw[(tau_lw >= p1l) & (tau_lw <= p99l)]
            kde_lw = gaussian_kde(tl, bw_method=0.1)
            xg_lw  = np.linspace(p1l, p99l, 400)
            ax.plot(xg_lw, kde_lw(xg_lw), color=lw_color,
                    lw=2.0, linestyle='--', label='LW')

        F_cm = results_cm[r]['flatness']
        K_cm = results_cm[r]['kurtosis']
        ax.set_title(f'$r={r}$\nF={F_cm:.2f}, K={K_cm:.2f}',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('$\\|\\tau\\|$', fontsize=8)
        ax.set_ylabel('Density', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused panels
    for ax in axes_flat[n_r:]:
        ax.axis('off')

    fig.suptitle(
        'Traction PDF at tip $(r,0,0)$ for each aspect ratio\n'
        'F = flatness, K = excess kurtosis  (Gaussian: F=3, K=0)\n'
        'High K = heavy tails = intermittent extreme traction events',
        fontsize=12, fontweight='bold'
    )
    fig.tight_layout()
    fig.savefig(OUTDIR + 'step12a_intermittency_pdfs.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved step12a_intermittency_pdfs.png")


def plot_flatness_kurtosis_vs_r(results_cm, results_lw=None, tau_K=None):
    """
    Flatness, excess kurtosis, skewness, mean traction, and
    non-parametric intermittency vs aspect ratio.
    2x3 grid — all 6 panels filled.
    """
    rs     = np.array(list(results_cm.keys()))
    F_cm   = np.array([results_cm[r]['flatness']      for r in rs])
    K_cm   = np.array([results_cm[r]['kurtosis']      for r in rs])
    F_err_cm = np.array([results_cm[r]['flatness_std'] for r in rs])
    K_err_cm = np.array([results_cm[r]['kurtosis_std'] for r in rs])
    sk_cm  = np.array([results_cm[r]['skewness']      for r in rs])
    mu_cm  = np.array([results_cm[r]['mean']          for r in rs])
    std_cm = np.array([results_cm[r]['std']           for r in rs])
    I_cm   = np.array([results_cm[r]['intermittency'] for r in rs])

    cm_color = '#E07B39'
    lw_color = '#4472C4'

    # LW arrays (only computed if results_lw is not None)
    if results_lw is not None:
        rs_lw  = np.array(list(results_lw.keys()))
        F_lw   = np.array([results_lw[r]['flatness']      for r in rs_lw])
        K_lw   = np.array([results_lw[r]['kurtosis']      for r in rs_lw])
        sk_lw  = np.array([results_lw[r]['skewness']      for r in rs_lw])
        mu_lw  = np.array([results_lw[r]['mean']          for r in rs_lw])
        std_lw = np.array([results_lw[r]['std']           for r in rs_lw])
        I_lw   = np.array([results_lw[r]['intermittency'] for r in rs_lw])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    def _add_lw(ax, y_lw, yerr_lw=None):
        if results_lw is not None:
            if yerr_lw is not None:
                ax.errorbar(rs_lw, y_lw, yerr=yerr_lw,
                            fmt='s--', color=lw_color,
                            lw=2.5, ms=7, capsize=4, label='LW')
            else:
                ax.plot(rs_lw, y_lw, 's--', color=lw_color,
                        lw=2.5, ms=7, label='LW')

    def _annotate_oblate_prolate(ax):
        """Add oblate/prolate text after ylim is set."""
        ylo, yhi = ax.get_ylim()
        ypos = ylo + 0.92 * (yhi - ylo)
        ax.text(0.35, ypos, 'Oblate', ha='center', fontsize=9, color='grey')
        ax.text(3.0,  ypos, 'Prolate', ha='center', fontsize=9, color='grey')

    # ── [0,0] Flatness ─────────────────────────────────────────
    ax = axes[0, 0]
    ax.errorbar(rs, F_cm, yerr=F_err_cm,
            fmt='o-', color=cm_color, lw=2.5, ms=7,
            capsize=4, label='CM')
    _add_lw(ax, F_lw if results_lw is not None else None,
            F_err_lw if results_lw is not None else None)
    ax.axhline(3.0, color='grey', lw=1.0, linestyle='--',
               label='Gaussian (F=3)')
    ax.axvline(1.0, color='grey', lw=0.8, linestyle=':')
    ax.set_xlabel('Aspect ratio $r = a/b$', fontsize=12)
    ax.set_ylabel('Flatness $F = \\langle\\tau^4\\rangle / \\langle\\tau^2\\rangle^2$',
                  fontsize=11)
    ax.set_title('Flatness vs aspect ratio\nF > 3: heavier tails than Gaussian',
                 fontsize=11, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    _annotate_oblate_prolate(ax)

    # ── [0,1] Excess kurtosis ──────────────────────────────────
    ax = axes[0, 1]
    ax.errorbar(rs, K_cm, yerr=K_err_cm,
                fmt='o-', color=cm_color, lw=2.5, ms=7,
                capsize=4, label='CM')
    _add_lw(ax, K_lw if results_lw is not None else None,
            K_err_lw if results_lw is not None else None)
    ax.axhline(0.0, color='grey', lw=1.0, linestyle='--',
               label='Gaussian (K=0)')
    ax.axvline(1.0, color='grey', lw=0.8, linestyle=':')
    ax.set_xlabel('Aspect ratio $r = a/b$', fontsize=12)
    ax.set_ylabel('Excess kurtosis $K = F - 3$', fontsize=11)
    ax.set_title('Excess kurtosis vs aspect ratio\nK > 0: more extreme events than Gaussian',
                 fontsize=11, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    _annotate_oblate_prolate(ax)

    # ── [0,2] Non-parametric intermittency ─────────────────────
    ax = axes[0, 2]
    ax.plot(rs, I_cm, 'o-', color=cm_color, lw=2.5, ms=7, label='CM')
    _add_lw(ax, I_lw if results_lw is not None else None)
    ax.axhline(1.0, color='grey', lw=1.0, linestyle='--',
               label='No intermittency')
    ax.axvline(1.0, color='grey', lw=0.8, linestyle=':')
    ax.set_xlabel('Aspect ratio $r = a/b$', fontsize=12)
    ax.set_ylabel('$P_{95} / P_{50}$', fontsize=12)
    ax.set_title('Non-parametric intermittency\n'
                 '$P_{95}/P_{50}$ — tail heaviness relative to median',
                 fontsize=11, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    _annotate_oblate_prolate(ax)

    # ── [1,0] Skewness ─────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(rs, sk_cm, 'o-', color=cm_color, lw=2.5, ms=7, label='CM')
    _add_lw(ax, sk_lw if results_lw is not None else None)
    ax.axhline(0.0, color='grey', lw=1.0, linestyle='--',
               label='Gaussian (skew=0)')
    ax.axvline(1.0, color='grey', lw=0.8, linestyle=':')
    ax.set_xlabel('Aspect ratio $r = a/b$', fontsize=12)
    ax.set_ylabel('Skewness', fontsize=11)
    ax.set_title('Skewness vs aspect ratio\n'
                 'Skew > 0: right tail (extreme high traction events)',
                 fontsize=11, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    _annotate_oblate_prolate(ax)

    # ── [1,1] Mean ± std ───────────────────────────────────────
    ax = axes[1, 1]
    ax.errorbar(rs, mu_cm, yerr=std_cm,
                fmt='o-', color=cm_color, lw=2.5, ms=7,
                capsize=4, label='CM  mean ± std')
    if results_lw is not None:
        ax.errorbar(rs_lw, mu_lw, yerr=std_lw,
                    fmt='s--', color=lw_color, lw=2.5, ms=7,
                    capsize=4, label='LW  mean ± std')
    ax.axvline(1.0, color='grey', lw=0.8, linestyle=':')
    ax.set_xlabel('Aspect ratio $r = a/b$', fontsize=12)
    ax.set_ylabel('$\\langle\\|\\tau\\|\\rangle$', fontsize=12)
    ax.set_title('Mean traction ± std vs aspect ratio',
                 fontsize=11, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    _annotate_oblate_prolate(ax)

    # ── [1,2] Coefficient of variation (std/mean) ──────────────
    # CV is scale-independent — shows relative variability vs r
    CV_cm = std_cm / (mu_cm + 1e-30)
    ax = axes[1, 2]
    ax.plot(rs, CV_cm, 'o-', color=cm_color, lw=2.5, ms=7, label='CM')
    if results_lw is not None:
        CV_lw = std_lw / (mu_lw + 1e-30)
        ax.plot(rs_lw, CV_lw, 's--', color=lw_color, lw=2.5,
                ms=7, label='LW')
    ax.axvline(1.0, color='grey', lw=0.8, linestyle=':')
    ax.set_xlabel('Aspect ratio $r = a/b$', fontsize=12)
    ax.set_ylabel('CV $= \\sigma / \\langle\\|\\tau\\|\\rangle$', fontsize=12)
    ax.set_title('Coefficient of variation vs aspect ratio\n'
                 'Scale-independent relative variability',
                 fontsize=11, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    _annotate_oblate_prolate(ax)

    fig.suptitle(
        'Traction intermittency vs aspect ratio\n'
        'All moments computed on normalised fluctuations $\\tau/\\langle\\tau\\rangle - 1$\n'
        'High F/K: particle shape amplifies turbulent fluctuations into extreme stresses',
        fontsize=12, fontweight='bold'
    )
    fig.tight_layout()
    fig.savefig(OUTDIR + 'step12a_flatness_kurtosis_vs_r.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved step12a_flatness_kurtosis_vs_r.png")


# ─────────────────────────────────────────────────────────────
# ANALYSIS 2 — TRACTION CONDITIONED ON Q-R SPACE
# ─────────────────────────────────────────────────────────────

def compute_QR_traction(A_series, tau_mag, n_bins=40):
    """
    Bin the (Q, R) plane and compute mean traction, std traction,
    and count in each bin.

    Returns:
      Q_edges, R_edges : bin edges
      mean_tau  : (n_bins, n_bins) mean traction per bin
      std_tau   : (n_bins, n_bins) std traction per bin
      count     : (n_bins, n_bins) number of samples per bin
      Q, R      : raw (T,) arrays for scatter plot
    """
    scalars = flow_scalars(A_series)
    Q       = scalars['Q']
    R       = scalars['R_inv']

    # Clip to central 99% to avoid extreme outliers dominating bins
    Q_lo, Q_hi = np.percentile(Q, 0.5), np.percentile(Q, 99.5)
    R_lo, R_hi = np.percentile(R, 0.5), np.percentile(R, 99.5)

    Q_edges = np.linspace(Q_lo, Q_hi, n_bins + 1)
    R_edges = np.linspace(R_lo, R_hi, n_bins + 1)

    mean_tau = np.full((n_bins, n_bins), np.nan)
    std_tau  = np.full((n_bins, n_bins), np.nan)
    count    = np.zeros((n_bins, n_bins), dtype=int)

    Q_idx = np.digitize(Q, Q_edges) - 1
    R_idx = np.digitize(R, R_edges) - 1

    # Clip indices to valid range
    Q_idx = np.clip(Q_idx, 0, n_bins - 1)
    R_idx = np.clip(R_idx, 0, n_bins - 1)

    for qi in range(n_bins):
        for ri in range(n_bins):
            mask = (Q_idx == qi) & (R_idx == ri)
            n    = mask.sum()
            if n >= 5:
                mean_tau[qi, ri] = np.mean(tau_mag[mask])
                std_tau[qi, ri]  = np.std(tau_mag[mask])
                count[qi, ri]    = n

    Q_centres = 0.5 * (Q_edges[:-1] + Q_edges[1:])
    R_centres = 0.5 * (R_edges[:-1] + R_edges[1:])

    return Q_edges, R_edges, Q_centres, R_centres, mean_tau, std_tau, count, Q, R


def plot_QR_traction(Q_edges, R_edges, Q_centres, R_centres,
                     mean_tau, std_tau, count, Q_raw, R_raw,
                     tau_raw, label='CM'):
    """
    Two-panel plot:
      Left:  Q-R plane coloured by mean traction (heatmap)
      Right: Q-R plane coloured by traction std (intermittency map)
    With the classic teardrop Q-R null discriminant curve overlaid.
    """
    cm_map   = 'inferno'
    std_map  = 'plasma'

    # Null discriminant: 27R^2 + 4Q^3 = 0  (boundary of real eigenvalues)
    R_line   = np.linspace(R_edges[0], R_edges[-1], 500)
    # Q = -(27/4 * R^2)^(1/3)  for the discriminant = 0 curve
    with np.errstate(invalid='ignore'):
        Q_disc = -np.cbrt(27.0/4.0 * R_line**2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ── Left: mean traction ────────────────────────────────────
    ax  = axes[0]
    vmin = np.nanpercentile(mean_tau, 5)
    vmax = np.nanpercentile(mean_tau, 95)
    im  = ax.pcolormesh(R_edges, Q_edges, mean_tau,
                        cmap=cm_map, vmin=vmin, vmax=vmax,
                        shading='flat')
    plt.colorbar(im, ax=ax, label='Mean $\\|\\tau\\|$')
    ax.plot(R_line, Q_disc, 'w--', lw=1.2, alpha=0.7,
            label='Discriminant $27R^2+4Q^3=0$')
    ax.axhline(0, color='white', lw=0.8, alpha=0.5)
    ax.axvline(0, color='white', lw=0.8, alpha=0.5)
    ax.set_xlabel('$R = -\\frac{1}{3}\\mathrm{Tr}(A^3)$', fontsize=12)
    ax.set_ylabel('$Q = \\frac{1}{2}(\\|W\\|^2 - \\|S\\|^2)$', fontsize=12)
    ax.set_title(f'[{label}] Mean traction in Q-R space\n'
                 'Q>0: rotation dominated,  Q<0: strain dominated',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')

    # ── Middle: traction std (intermittency) ───────────────────
    ax  = axes[1]
    vmin_s = np.nanpercentile(std_tau, 5)
    vmax_s = np.nanpercentile(std_tau, 95)
    im2 = ax.pcolormesh(R_edges, Q_edges, std_tau,
                        cmap=std_map, vmin=vmin_s, vmax=vmax_s,
                        shading='flat')
    plt.colorbar(im2, ax=ax, label='Std $\\|\\tau\\|$')
    ax.plot(R_line, Q_disc, 'w--', lw=1.2, alpha=0.7)
    ax.axhline(0, color='white', lw=0.8, alpha=0.5)
    ax.axvline(0, color='white', lw=0.8, alpha=0.5)
    ax.set_xlabel('$R$', fontsize=12)
    ax.set_ylabel('$Q$', fontsize=12)
    ax.set_title(f'[{label}] Traction std in Q-R space\n'
                 'High std = intermittent traction in that flow region',
                 fontsize=11, fontweight='bold')

    # ── Right: sample count (shows where Q-R is sampled) ───────
    ax  = axes[2]
    im3 = ax.pcolormesh(R_edges, Q_edges, np.log1p(count),
                        cmap='viridis', shading='flat')
    plt.colorbar(im3, ax=ax, label='$\\log(1 + N)$')
    ax.plot(R_line, Q_disc, 'w--', lw=1.2, alpha=0.7,
            label='Discriminant')
    ax.axhline(0, color='white', lw=0.8, alpha=0.5)
    ax.axvline(0, color='white', lw=0.8, alpha=0.5)
    ax.set_xlabel('$R$', fontsize=12)
    ax.set_ylabel('$Q$', fontsize=12)
    ax.set_title(f'[{label}] Sample density in Q-R space\n'
                 'Expected teardrop shape for turbulence',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)

    fig.suptitle(
        f'[{label}] Traction conditioned on Q-R invariant space\n'
        'Which velocity gradient configurations produce the most dangerous surface stress?',
        fontsize=13, fontweight='bold'
    )
    fig.tight_layout()
    fname = OUTDIR + f'step12b_QR_traction_{label.lower()}.png'
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {fname}")


def plot_QR_traction_scatter(Q_raw, R_raw, tau_raw, label='CM',
                              n_scatter=5000):
    """
    Scatter plot of (R, Q) coloured by traction magnitude.
    Uses a random subsample for visibility.
    """
    idx = np.random.choice(len(Q_raw), size=min(n_scatter, len(Q_raw)),
                           replace=False)
    Q_s   = Q_raw[idx]
    R_s   = R_raw[idx]
    tau_s = tau_raw[idx]

    # Null discriminant
    R_line = np.linspace(np.percentile(R_raw, 0.5),
                         np.percentile(R_raw, 99.5), 500)
    with np.errstate(invalid='ignore'):
        Q_disc = -np.cbrt(27.0/4.0 * R_line**2)

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(R_s, Q_s, c=tau_s, cmap='inferno',
                    s=4, alpha=0.6,
                    vmin=np.percentile(tau_raw, 5),
                    vmax=np.percentile(tau_raw, 95))
    plt.colorbar(sc, ax=ax, label='$\\|\\tau\\|$')
    ax.plot(R_line, Q_disc, 'k--', lw=1.5, alpha=0.8,
            label='Discriminant $27R^2+4Q^3=0$')
    ax.axhline(0, color='grey', lw=0.8, alpha=0.6)
    ax.axvline(0, color='grey', lw=0.8, alpha=0.6)

    # Label the four quadrants
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    pad  = 0.05
    ax.text(xlim[1]*(1-pad), ylim[1]*(1-pad),
            'Unstable\nfocus', ha='right', va='top',
            fontsize=8, color='white', alpha=0.8)
    ax.text(xlim[0]*(1-pad), ylim[1]*(1-pad),
            'Stable\nfocus', ha='left', va='top',
            fontsize=8, color='white', alpha=0.8)
    ax.text(xlim[1]*(1-pad), ylim[0]*(1-pad),
            'Unstable\nnode', ha='right', va='bottom',
            fontsize=8, color='white', alpha=0.8)
    ax.text(xlim[0]*(1-pad), ylim[0]*(1-pad),
            'Stable\nnode', ha='left', va='bottom',
            fontsize=8, color='white', alpha=0.8)

    ax.set_xlabel('$R = -\\frac{1}{3}\\mathrm{Tr}(A^3)$', fontsize=12)
    ax.set_ylabel('$Q = \\frac{1}{2}(\\|W\\|^2 - \\|S\\|^2)$', fontsize=12)
    ax.set_title(
        f'[{label}] Q-R scatter coloured by traction $\\|\\tau\\|$\n'
        f'({n_scatter} random samples from {len(Q_raw)} total)',
        fontsize=11, fontweight='bold'
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fname = OUTDIR + f'step12b_QR_traction_scatter_{label.lower()}.png'
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {fname}")


def plot_QR_comparison(qr_cm, qr_lw):
    """
    Side-by-side mean traction in Q-R for CM vs LW.
    Uses a shared colour scale for direct comparison.
    """
    _, _, _, _, mean_cm, _, _, _, _ = qr_cm
    _, _, _, _, mean_lw, _, _, _, _ = qr_lw

    vmin = min(np.nanpercentile(mean_cm, 5), np.nanpercentile(mean_lw, 5))
    vmax = max(np.nanpercentile(mean_cm, 95), np.nanpercentile(mean_lw, 95))

    Q_edges_cm, R_edges_cm = qr_cm[0], qr_cm[1]
    Q_edges_lw, R_edges_lw = qr_lw[0], qr_lw[1]

    R_line = np.linspace(min(R_edges_cm[0], R_edges_lw[0]),
                         max(R_edges_cm[-1], R_edges_lw[-1]), 500)
    with np.errstate(invalid='ignore'):
        Q_disc = -np.cbrt(27.0/4.0 * R_line**2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, mean_t, Q_e, R_e, lbl in zip(
            axes[:2],
            [mean_cm, mean_lw],
            [Q_edges_cm, Q_edges_lw],
            [R_edges_cm, R_edges_lw],
            ['CM', 'LW']):
        im = ax.pcolormesh(R_e, Q_e, mean_t, cmap='inferno',
                           vmin=vmin, vmax=vmax, shading='flat')
        plt.colorbar(im, ax=ax, label='Mean $\\|\\tau\\|$')
        ax.plot(R_line, Q_disc, 'w--', lw=1.2, alpha=0.7)
        ax.axhline(0, color='white', lw=0.8, alpha=0.5)
        ax.axvline(0, color='white', lw=0.8, alpha=0.5)
        ax.set_xlabel('$R$', fontsize=12)
        ax.set_ylabel('$Q$', fontsize=12)
        ax.set_title(f'[{lbl}] Mean traction in Q-R space',
                     fontsize=11, fontweight='bold')

    # Difference panel
    ax = axes[2]
    # Interpolate to common grid if needed — use CM grid
    diff = mean_lw - mean_cm
    vd   = np.nanpercentile(np.abs(diff), 95)
    im3  = ax.pcolormesh(R_edges_cm, Q_edges_cm, diff,
                         cmap='RdBu_r', vmin=-vd, vmax=vd,
                         shading='flat')
    plt.colorbar(im3, ax=ax, label='LW - CM')
    ax.plot(R_line, Q_disc, 'k--', lw=1.2, alpha=0.7)
    ax.axhline(0, color='grey', lw=0.8, alpha=0.5)
    ax.axvline(0, color='grey', lw=0.8, alpha=0.5)
    ax.set_xlabel('$R$', fontsize=12)
    ax.set_ylabel('$Q$', fontsize=12)
    ax.set_title('Difference LW - CM\n'
                 'Shows where turbulence model choice matters most',
                 fontsize=11, fontweight='bold')

    fig.suptitle(
        'Q-R traction comparison: CM vs LW\n'
        'Shared colour scale — direct comparison of mean traction\n'
        'in different velocity gradient configurations',
        fontsize=13, fontweight='bold'
    )
    fig.tight_layout()
    fig.savefig(OUTDIR + 'step12b_QR_comparison_CM_LW.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved step12b_QR_comparison_CM_LW.png")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':

    print("── Step 12: Traction Intermittency + Q-R Analysis ──\n")

    # ── Load CM ─────────────────────────────────────────────────
    print("Loading CM data...")
    times_cm, A_cm = load_csv(CSV_PATH)

    print("Integrating orientation ODE (CM)...")
    R_cm, *_ = integrate_orientation(AXES, A_cm, times_cm)

    times_b, A_b, R_b = apply_burnin(times_cm, A_cm, R_cm, BURN_IN)
    tau_K = kolmogorov_time(A_b)
    print(f"  tau_K = {tau_K:.4f}")

    # ── Load LW (optional) ─────────────────────────────────────
    A_lw_b = None; R_lw_b = None; tau_lw_b = None

    if LW_CSV_PATH is not None and os.path.exists(LW_CSV_PATH):
        print(f"\nLoading LW data from '{LW_CSV_PATH}'...")
        times_lw, A_lw = load_csv(LW_CSV_PATH)
        print("Integrating orientation ODE (LW)...")
        R_lw, *_ = integrate_orientation(AXES, A_lw, times_lw)
        _, A_lw_b, R_lw_b = apply_burnin(times_lw, A_lw, R_lw, BURN_IN)
    elif LW_CSV_PATH is not None:
        print(f"  '{LW_CSV_PATH}' not found — running CM only.")

    # ── 12a: Intermittency vs aspect ratio ──────────────────────
    print("\n── 12a: Intermittency vs aspect ratio ──")
    print("  Computing traction for each aspect ratio (CM)...")
    results_cm = compute_intermittency_vs_r(A_b, R_b, ASPECT_RATIOS, times_b, mu=MU)

    results_lw = None
    if A_lw_b is not None:
        print("  Computing traction for each aspect ratio (LW)...")
        results_lw = compute_intermittency_vs_r(
            A_lw_b, R_lw_b, ASPECT_RATIOS, times_b, mu=MU)

    plot_intermittency_pdfs(results_cm, results_lw)
    plot_flatness_kurtosis_vs_r(results_cm, results_lw, tau_K=tau_K)

    # ── 12b: Q-R conditioned traction ───────────────────────────
    print("\n── 12b: Q-R conditioned traction ──")

    # Use AXES traction at tip for Q-R analysis
    x_tip = np.array([AXES[0], 0., 0.])
    print("  Building transfer matrix (CM)...")
    M_cm, n_hat_cm = build_transfer_matrix(AXES, x_tip, mu=MU)

    print("  Computing traction time series (CM, fast)...")
    tau_cm = fast_traction_magnitude(M_cm, n_hat_cm, A_b, R_b)

    print("  Binning Q-R plane (CM)...")
    qr_cm = compute_QR_traction(A_b, tau_cm, n_bins=N_QR_BINS)
    Q_edges, R_edges, Q_c, R_c, mean_tau, std_tau, count, Q_raw, R_raw = qr_cm

    plot_QR_traction(Q_edges, R_edges, Q_c, R_c,
                     mean_tau, std_tau, count, Q_raw, R_raw,
                     tau_cm, label='CM')
    plot_QR_traction_scatter(Q_raw, R_raw, tau_cm, label='CM')

    if A_lw_b is not None:
        print("  Building transfer matrix (LW — same geometry as CM)...")
        # M is the same since axes and surface point are identical
        print("  Computing traction time series (LW, fast)...")
        tau_lw = fast_traction_magnitude(M_cm, n_hat_cm, A_lw_b, R_lw_b)

        print("  Binning Q-R plane (LW)...")
        qr_lw = compute_QR_traction(A_lw_b, tau_lw, n_bins=N_QR_BINS)
        Q_e_lw, R_e_lw, Q_c_lw, R_c_lw, mt_lw, st_lw, cnt_lw, Q_lw, R_lw = qr_lw

        plot_QR_traction(Q_e_lw, R_e_lw, Q_c_lw, R_c_lw,
                         mt_lw, st_lw, cnt_lw, Q_lw, R_lw,
                         tau_lw, label='LW')
        plot_QR_traction_scatter(Q_lw, R_lw, tau_lw, label='LW')
        plot_QR_comparison(qr_cm, qr_lw)

    print(f"\nAll step 12 plots saved to '{OUTDIR}'")
    print("Done.")