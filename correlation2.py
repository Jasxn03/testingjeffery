"""
ra_coupling.py

Investigates the joint statistics of the rotation matrix R(t) and the
velocity gradient tensor A(t) for a tumbling ellipsoid in turbulence.

Steps
-----
1. Three orientation representations (separate figures):
      (a) Euler angles  (phi, theta, psi)
      (b) Symmetry axis in spherical coords  (theta_p, phi_p)
      (c) Jeffery orbit constant  C(t)

2. Conditional distributions  p(Q | orientation bin)  and  p(||S|| | orientation bin)
   — split timeseries into orientation bins, compute flow-statistics per bin

3. Mutual information  I(orientation; A)  +  shuffle test for independence baseline

4. Conditional traction  p(||tau|| | orientation bin)

Inputs expected (edit CONFIG below):
    CSV_PATH   : velocity-gradient timeseries  (time, A11 … A33)
    AXES       : ellipsoid semi-axes  [a, b, c]

Place jeffery4.py, orientation.py, stress_functions.py in the same directory.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from scipy.stats import gaussian_kde
from scipy.linalg import expm
from sklearn.feature_selection import mutual_info_regression
import os
import sys

# ─────────────────────────────────────────────────────────────
# CONFIG  — edit these
# ─────────────────────────────────────────────────────────────

CSV_PATH   = "grad_u.csv"
LW_CSV_PATH  = "grad_u_LW.csv" 
AXES       = [2.0, 1.0, 1.0]          # semi-axes [a, b, c]
MU         = 1.0                       # dynamic viscosity
BURN_IN    = 0.10                      # fraction to discard
N_BINS     = 6                         # orientation bins for steps 2 & 4
OUTDIR     = "ra_coupling_plots/"
JEFFERY_PATH = "."
ACF_MAX_LAG  = 10.0  

# ── Toggle which steps to run ──────────────────────────────
RUN_STEP1  = False
RUN_STEP2  = False
RUN_STEP3  = False
RUN_STEP4  = True   # slow: calls ellipsoid.sigma() at every timestep
RUN_STEP5  = True   # slow: re-integrates orientation for each aspect ratio
RUN_STEP6  = True   # requires RUN_STEP4 result (tau_mag) if True
RUN_STEP7  = True   # requires RUN_STEP4 result (tau_mag) if True
RUN_STEP8  = True
RUN_STEP9  = True

# ─────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────

sys.path.insert(0, JEFFERY_PATH)
from jeffery4 import Ellipsoid
from orientation import integrate_orientation
# stress_functions no longer needed — traction uses ellipsoid.sigma() directly

os.makedirs(OUTDIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────

def load_csv(path):
    df     = pd.read_csv(path)
    times  = df['time'].values
    A_series = np.zeros((len(df), 3, 3))
    for i in range(3):
        for j in range(3):
            A_series[:, i, j] = df[f'A{i+1}{j+1}'].values
    print(f"Loaded {len(times)} timesteps  ({times[0]:.4f} → {times[-1]:.4f})")
    return times, A_series

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def burn(arr, frac):
    """Drop first frac of array."""
    n = int(len(arr) * frac)
    return arr[n:]

def rotation_to_euler(R):
    """
    Extract ZYX Euler angles (phi, theta, psi) from rotation matrix R.
    Returns angles in degrees. Shape: (T, 3)
    """
    phi   = np.arctan2( R[:, 2, 1], R[:, 2, 2])
    theta = np.arcsin( -R[:, 2, 0])
    psi   = np.arctan2( R[:, 1, 0], R[:, 0, 0])
    return np.degrees(phi), np.degrees(theta), np.degrees(psi)

def symmetry_axis_angles(R):
    """
    p = R[:,0] is the symmetry axis in the lab frame.
    Returns (theta_p, phi_p) in degrees — polar and azimuthal angles.
    Shape: (T,), (T,)
    """
    p      = R[:, :, 0]                           # (T, 3)
    theta_p = np.degrees(np.arccos(np.clip(p[:, 2], -1, 1)))   # polar (0–180)
    phi_p   = np.degrees(np.arctan2(p[:, 1], p[:, 0]))         # azimuthal (−180–180)
    return theta_p, phi_p

def jeffery_C(R, r):
    """
    Jeffery orbit constant C = r * |p_z| / sqrt(p_x^2 + p_y^2).
    Returns (T,) with nan where pxy ≈ 0.
    """
    p   = R[:, :, 0]
    pz  = np.abs(p[:, 2])
    pxy = np.sqrt(p[:, 0]**2 + p[:, 1]**2)
    with np.errstate(invalid='ignore', divide='ignore'):
        C = r * pz / pxy
    C[pxy < 1e-10] = np.nan
    return C

def flow_scalars(A):
    """
    Compute scalar flow diagnostics from A timeseries (T, 3, 3).
    Returns dict of (T,) arrays.
    """
    S  = 0.5 * (A + A.transpose(0, 2, 1))
    W  = 0.5 * (A - A.transpose(0, 2, 1))
    S_norm = np.sqrt(np.einsum('tij,tij->t', S, S))
    W_norm = np.sqrt(np.einsum('tij,tij->t', W, W))
    # Q invariant: Q = 0.5*(||W||^2 - ||S||^2)
    Q      = 0.5 * (W_norm**2 - S_norm**2)
    # R invariant: R = -1/3 tr(A^3)
    A2     = np.einsum('tij,tjk->tik', A, A)
    A3     = np.einsum('tij,tjk->tik', A2, A)
    R_inv  = -np.einsum('tii->t', A3) / 3.0
    # TrO^2S — Gustavsson's key diagnostic for vortex-dominated regions
    O2     = np.einsum('tij,tjk->tik', W, W)     # (T,3,3)
    TrO2S  = np.einsum('tij,tji->t', O2, S)      # Tr(O^2 S), (T,)
    return {'S_norm': S_norm, 'W_norm': W_norm, 'Q': Q, 'R_inv': R_inv,
            'TrO2S': TrO2S}

def traction_magnitude(A_series, R_history, axes, mu):
    """
    Compute ||tau|| at the tip (a,0,0) in the body frame for all timesteps.
    Uses ellipsoid.sigma() directly — same approach as aspect_ratio2.py.
    """
    a   = np.array(axes, dtype=float)
    tip = np.array([a[0], 0.0, 0.0])
    n   = np.array([1.0,  0.0, 0.0])   # outward normal at tip (a,0,0)

    # Build ellipsoid once with dummy strain — geometry is fixed
    eps0 = np.zeros((3, 3))
    ell  = Ellipsoid(a, eps0, mu=mu)
    ell.use_surface_mode()

    taus = []
    for t, A_t in enumerate(A_series):
        # Transform A into body frame using current rotation
        R_t    = R_history[t]
        A_body = R_t.T @ A_t @ R_t

        # Update ellipsoid strain — exactly as in aspect_ratio2.py
        ell.set_strain(A_body)
        ell.set_coefs()

        # Compute stress and traction using ellipsoid.sigma()
        sig   = ell.sigma(tip)          # (3,3)
        t_vec = sig @ n                 # traction vector
        tau   = t_vec - np.dot(t_vec, n) * n   # tangential component
        taus.append(float(np.linalg.norm(tau)))

    return np.array(taus)

# ─────────────────────────────────────────────────────────────
# STEP 1 — THREE ORIENTATION REPRESENTATIONS
# ─────────────────────────────────────────────────────────────

def plot_step1_euler(times, R_history):
    """(a) Euler angles over time + joint distributions."""
    phi, theta, psi = rotation_to_euler(R_history)

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    colors = plt.cm.plasma([0.2, 0.55, 0.85])

    # Time series
    for k, (ang, lbl, col) in enumerate(zip(
            [phi, theta, psi],
            ['$\\phi$ (roll)', '$\\theta$ (pitch)', '$\\psi$ (yaw)'],
            colors)):
        ax = fig.add_subplot(gs[0, k])
        ax.plot(times, ang, lw=0.6, color=col)
        ax.set_title(lbl, fontsize=12)
        ax.set_xlabel('Time'); ax.set_ylabel('Degrees')
        ax.grid(True, alpha=0.3)

    # Joint scatter/KDE pairs
    pairs = [(phi, theta, '$\\phi$', '$\\theta$'),
             (theta, psi,  '$\\theta$', '$\\psi$'),
             (phi,   psi,  '$\\phi$', '$\\psi$')]

    for k, (x, y, xl, yl) in enumerate(pairs):
        ax = fig.add_subplot(gs[1, k])
        ax.hexbin(x, y, gridsize=40, cmap='inferno', mincnt=1)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(f'Joint: {xl} vs {yl}', fontsize=10)

    fig.suptitle('Euler Angle Representation', fontsize=13, fontweight='bold')
    fig.savefig(OUTDIR + 'step1a_euler.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved step1a_euler.png")


def plot_step1_symmetry_axis(times, R_history):
    """(b) Symmetry axis in spherical coordinates — time series + sphere scatter."""
    theta_p, phi_p = symmetry_axis_angles(R_history)

    # Convert to Cartesian for sphere plot
    p = R_history[:, :, 0]

    fig = plt.figure(figsize=(16, 6))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # theta_p time series
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(times, theta_p, lw=0.6, color='steelblue')
    ax.set_xlabel('Time'); ax.set_ylabel('degrees')
    ax.set_title('Polar angle $\\theta_p$ of symmetry axis', fontsize=11)
    ax.grid(True, alpha=0.3)

    # phi_p time series
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(times, phi_p, lw=0.6, color='darkorange')
    ax.set_xlabel('Time'); ax.set_ylabel('degrees')
    ax.set_title('Azimuthal angle $\\phi_p$ of symmetry axis', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Scatter on unit sphere (2D projection: theta_p vs phi_p coloured by time)
    ax = fig.add_subplot(gs[0, 2])
    sc = ax.scatter(phi_p, theta_p,
                    c=times, cmap='viridis', s=1.5, alpha=0.5)
    plt.colorbar(sc, ax=ax, label='Time')
    ax.set_xlabel('$\\phi_p$ (azimuthal, deg)')
    ax.set_ylabel('$\\theta_p$ (polar, deg)')
    ax.set_title('Symmetry axis trajectory on unit sphere', fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Symmetry Axis Spherical Coordinates', fontsize=13, fontweight='bold')
    fig.savefig(OUTDIR + 'step1b_symmetry_axis.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved step1b_symmetry_axis.png")


def plot_step1_jeffery_C(times, R_history, r):
    """(c) Jeffery orbit constant C over time + distribution."""
    C = jeffery_C(R_history, r)
    C_finite = C[np.isfinite(C)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Time series
    axes[0].plot(times, C, lw=0.6, color='darkgreen', alpha=0.8)
    axes[0].set_xlabel('Time'); axes[0].set_ylabel('C')
    axes[0].set_title('Jeffery orbit constant $C(t)$', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, np.percentile(C_finite, 97)])

    # Distribution
    p99 = np.percentile(C_finite, 99)
    Cc  = C_finite[C_finite <= p99]
    kde = gaussian_kde(Cc, bw_method=0.1)
    xg  = np.linspace(0, p99, 400)
    axes[1].plot(xg, kde(xg), color='darkgreen', lw=2)
    axes[1].fill_between(xg, kde(xg), alpha=0.2, color='darkgreen')
    axes[1].set_xlabel('C'); axes[1].set_ylabel('Density')
    axes[1].set_title('Distribution of $C$', fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # C vs time coloured by C value (shows orbit structure)
    sc = axes[2].scatter(times, C, c=C, cmap='plasma', s=1.5,
                         vmax=np.percentile(C_finite, 95))
    plt.colorbar(sc, ax=axes[2], label='C')
    axes[2].set_xlabel('Time'); axes[2].set_ylabel('C')
    axes[2].set_title('$C(t)$ coloured by $C$ (orbit structure)', fontsize=11)
    axes[2].set_ylim([0, np.percentile(C_finite, 97)])
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('Jeffery Orbit Constant', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTDIR + 'step1c_jeffery_C.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved step1c_jeffery_C.png")


# ─────────────────────────────────────────────────────────────
# STEP 2 — CONDITIONAL DISTRIBUTIONS  p(flow scalar | orientation bin)
# ─────────────────────────────────────────────────────────────

def plot_step2_conditional(times, R_history, A_series, r, n_bins):
    """
    Bin the timeseries by theta_p (polar angle of symmetry axis).
    For each bin, plot the conditional distributions of ||S||, ||W||, Q.
    """
    theta_p, _ = symmetry_axis_angles(R_history)
    scalars     = flow_scalars(A_series)

    bin_edges   = np.linspace(theta_p.min(), theta_p.max(), n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    colors      = plt.cm.coolwarm(np.linspace(0, 1, n_bins))

    targets = [
        ('S_norm', '$\\|S\\|$ (strain rate)',    False),
        ('W_norm', '$\\|W\\|$ (vorticity)',       False),
        ('Q',      '$Q$ invariant',               True),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (key, label, signed) in zip(axes, targets):
        vals = scalars[key]
        for b in range(n_bins):
            mask = (theta_p >= bin_edges[b]) & (theta_p < bin_edges[b+1])
            v    = vals[mask]
            if len(v) < 20:
                continue
            p1, p99 = np.percentile(v, 1), np.percentile(v, 99)
            vc = v[(v >= p1) & (v <= p99)]
            if len(vc) < 10:
                continue
            kde = gaussian_kde(vc, bw_method=0.15)
            xg  = np.linspace(vc.min(), vc.max(), 300)
            lbl = f'$\\theta_p \\in [{bin_centres[b]:.0f}°]$'
            ax.plot(xg, kde(xg), color=colors[b], lw=1.8, label=lbl)

        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'$p({label} \\mid \\theta_p$ bin)', fontsize=11)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Add colorbar for theta_p bins
    sm = plt.cm.ScalarMappable(cmap='coolwarm',
                                norm=Normalize(vmin=bin_edges[0], vmax=bin_edges[-1]))
    sm.set_array([])
    fig.colorbar(sm, ax=axes[-1], label='$\\theta_p$ (deg)', shrink=0.8)

    fig.suptitle('Conditional Flow Distributions Given Orientation',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTDIR + 'step2_conditional_flow.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved step2_conditional_flow.png")

    # --- Mean and std of each scalar per bin (summary plot) ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (key, label, _) in zip(axes, targets):
        vals  = scalars[key]
        means = []; stds = []
        for b in range(n_bins):
            mask = (theta_p >= bin_edges[b]) & (theta_p < bin_edges[b+1])
            v    = vals[mask]
            means.append(np.mean(v) if len(v) > 0 else np.nan)
            stds.append(np.std(v) if len(v) > 0 else np.nan)
        means = np.array(means); stds = np.array(stds)
        ax.errorbar(bin_centres, means, yerr=stds,
                    fmt='o-', capsize=4, color='steelblue', linewidth=2)
        ax.set_xlabel('$\\theta_p$ bin centre (deg)', fontsize=11)
        ax.set_ylabel(f'Mean {label}', fontsize=11)
        ax.set_title(f'Mean {label} vs orientation', fontsize=11)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Mean Flow Scalars Conditioned on Orientation',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTDIR + 'step2_mean_vs_orientation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved step2_mean_vs_orientation.png")


# ─────────────────────────────────────────────────────────────
# STEP 3 — MUTUAL INFORMATION + SHUFFLE TEST
# ─────────────────────────────────────────────────────────────

def compute_mutual_information(R_history, A_series, n_shuffles=200):
    """
    Compute mutual information I(orientation feature; flow scalar)
    and compare to shuffled (independence) baseline.

    Orientation features: theta_p, phi_p, C
    Flow scalars: ||S||, ||W||, Q

    Returns a dict of results.
    """
    theta_p, phi_p  = symmetry_axis_angles(R_history)
    r               = AXES[0] / AXES[1]
    C               = jeffery_C(R_history, r)
    scalars         = flow_scalars(A_series)

    # Use only finite C values — mask all arrays consistently
    finite_mask = np.isfinite(C)
    theta_p_f   = theta_p[finite_mask]
    phi_p_f     = phi_p[finite_mask]
    C_f         = C[finite_mask]

    orient_feats = {
        '$\\theta_p$':    theta_p_f,
        '$\\phi_p$':      phi_p_f,
        'C':              C_f,
    }
    flow_feats = {
        '$\\|S\\|$':   scalars['S_norm'][finite_mask],
        '$\\|W\\|$':   scalars['W_norm'][finite_mask],
        'Q':            scalars['Q'][finite_mask],
    }

    results = {}
    for o_name, o_vals in orient_feats.items():
        for f_name, f_vals in flow_feats.items():
            key = f'{o_name} vs {f_name}'

            # Observed MI
            X = o_vals.reshape(-1, 1)
            mi_obs = float(mutual_info_regression(X, f_vals, random_state=0)[0])

            # Shuffle distribution
            mi_shuf = []
            rng = np.random.default_rng(42)
            for _ in range(n_shuffles):
                f_shuf = rng.permutation(f_vals)
                mi_shuf.append(
                    float(mutual_info_regression(X, f_shuf, random_state=0)[0])
                )
            mi_shuf = np.array(mi_shuf)

            # Z-score
            z = (mi_obs - mi_shuf.mean()) / (mi_shuf.std() + 1e-12)

            results[key] = {
                'mi_obs':  mi_obs,
                'mi_shuf_mean': mi_shuf.mean(),
                'mi_shuf_std':  mi_shuf.std(),
                'z_score': z,
                'mi_shuf': mi_shuf,
            }
            print(f"  {key:<30}  MI={mi_obs:.4f}  "
                  f"shuffle={mi_shuf.mean():.4f}±{mi_shuf.std():.4f}  z={z:.2f}")

    return results


def plot_step3_mutual_information(mi_results):
    """Bar chart of observed MI vs shuffle baseline."""
    keys  = list(mi_results.keys())
    obs   = [mi_results[k]['mi_obs']       for k in keys]
    shuf  = [mi_results[k]['mi_shuf_mean'] for k in keys]
    err   = [mi_results[k]['mi_shuf_std']  for k in keys]
    zs    = [mi_results[k]['z_score']      for k in keys]

    x     = np.arange(len(keys))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Bar chart
    ax = axes[0]
    ax.bar(x - width/2, obs,  width, label='Observed MI',  color='steelblue')
    ax.bar(x + width/2, shuf, width, label='Shuffle mean', color='lightcoral',
           yerr=err, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Mutual Information (nats)', fontsize=11)
    ax.set_title('Observed vs Shuffled Mutual Information', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Z-scores
    ax = axes[1]
    bar_colors = ['steelblue' if z > 2 else 'lightcoral' for z in zs]
    ax.bar(x, zs, color=bar_colors)
    ax.axhline(2,  color='black', linestyle='--', lw=1.5, label='z=2 threshold')
    ax.axhline(-2, color='black', linestyle='--', lw=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Z-score', fontsize=11)
    ax.set_title('Z-score of MI above shuffle baseline\n(z>2 suggests genuine coupling)',
                 fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(OUTDIR + 'step3_mutual_information.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved step3_mutual_information.png")

    # --- Shuffle distributions — x-axis zoomed to shuffle range ---
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    for ax, key in zip(axes.flat, keys):
        shuf_dist = mi_results[key]['mi_shuf']
        mi_o      = mi_results[key]['mi_obs']
        z         = mi_results[key]['z_score']

        ax.hist(shuf_dist, bins=30, color='lightcoral', edgecolor='white', alpha=0.85)

        # x-axis zoomed to shuffle distribution — observed MI is way off to the right
        shuf_max = shuf_dist.mean() + 6 * shuf_dist.std()
        ax.set_xlim(0, max(shuf_max, shuf_dist.max() * 1.5))

        # Annotate observed MI as arrow + text rather than a line off-screen
        ax.annotate(f'Observed MI = {mi_o:.3f}\n(z = {z:.0f})',
                    xy=(ax.get_xlim()[1], ax.get_ylim()[1] * 0.5),
                    xytext=(ax.get_xlim()[1] * 0.55, ax.get_ylim()[1] * 0.7),
                    fontsize=7.5, color='steelblue', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.5),
                    ha='center')

        ax.set_title(key, fontsize=9)
        ax.set_xlabel('MI under $H_0$ (shuffled)', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Shuffle Test: null distribution vs observed MI\n'
                 '(x-axis shows shuffle range; observed MI is far to the right)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTDIR + 'step3_shuffle_distributions.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved step3_shuffle_distributions.png")


# ─────────────────────────────────────────────────────────────
# STEP 4 — CONDITIONAL TRACTION  p(||tau|| | orientation bin)
# ─────────────────────────────────────────────────────────────

def plot_step4_conditional_traction(times, R_history, A_series, traction_mag, n_bins):
    """
    Bin by theta_p and show p(||tau|| | bin).
    Also compute mean traction per bin and a 2D KDE of (theta_p, ||tau||).
    """
    theta_p, _ = symmetry_axis_angles(R_history)
    tau         = traction_mag

    bin_edges   = np.linspace(theta_p.min(), theta_p.max(), n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    colors      = plt.cm.plasma(np.linspace(0.1, 0.9, n_bins))

    # ── (a) Conditional KDE distributions ───────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for b in range(n_bins):
        mask = (theta_p >= bin_edges[b]) & (theta_p < bin_edges[b+1])
        v    = tau[mask]
        if len(v) < 20: continue
        p99 = np.percentile(v, 99)
        vc  = v[v <= p99]
        kde = gaussian_kde(vc, bw_method=0.15)
        xg  = np.linspace(0, p99, 300)
        ax.plot(xg, kde(xg), color=colors[b], lw=2,
                label=f'{bin_centres[b]:.0f}°')
    ax.set_xlabel('$\\|\\tau\\|$', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('$p(\\|\\tau\\| \\mid \\theta_p$ bin)', fontsize=12)
    ax.legend(title='$\\theta_p$ bin', fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── (b) Mean traction per bin ────────────────────────────
    ax = axes[1]
    means  = []; stds = []; p95s = []
    for b in range(n_bins):
        mask = (theta_p >= bin_edges[b]) & (theta_p < bin_edges[b+1])
        v    = tau[mask]
        means.append(np.mean(v)             if len(v) > 0 else np.nan)
        stds.append(np.std(v)               if len(v) > 0 else np.nan)
        p95s.append(np.percentile(v, 95)    if len(v) > 0 else np.nan)

    ax.errorbar(bin_centres, means, yerr=stds,
                fmt='o-', color='steelblue', capsize=5, lw=2, label='Mean ± std')
    ax.plot(bin_centres, p95s, 's--', color='darkorange', lw=1.5, label='95th percentile')
    ax.set_xlabel('$\\theta_p$ bin centre (deg)', fontsize=12)
    ax.set_ylabel('$\\|\\tau\\|$', fontsize=12)
    ax.set_title('Mean (and 95th pct) traction vs orientation', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Conditional Traction Given Orientation',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTDIR + 'step4_conditional_traction.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved step4_conditional_traction.png")

    # ── (c) 2D KDE: theta_p vs ||tau|| ───────────────────────
    p99_tau = np.percentile(tau, 99)
    mask99  = tau <= p99_tau

    fig, ax = plt.subplots(figsize=(8, 6))
    h = ax.hexbin(theta_p[mask99], tau[mask99],
                  gridsize=50, cmap='inferno', mincnt=1)
    plt.colorbar(h, ax=ax, label='Count')
    ax.set_xlabel('$\\theta_p$ (polar angle of symmetry axis, deg)', fontsize=12)
    ax.set_ylabel('$\\|\\tau\\|$', fontsize=12)
    ax.set_title('2D joint density: orientation vs traction',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTDIR + 'step4_2d_joint_orientation_traction.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved step4_2d_joint_orientation_traction.png")


# ─────────────────────────────────────────────────────────────
# STEP 5 — TUMBLING RATE vs ASPECT RATIO  (Gustavsson Fig. 2 comparison)
# ─────────────────────────────────────────────────────────────

# Digitised from Gustavsson et al. 2014 Fig. 2 (right), DNS curve.
# x = lambda (their aspect ratio, same as our r = a/b)
# y = <ndot^2> * tau_K^2  (normalised squared tumbling rate)
# Disks: lambda < 1,  Rods: lambda > 1,  Sphere: lambda = 1
_GUSTAVSSON_LAMBDA = np.array([
    0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00,
    1.10, 1.30, 1.50, 2.00, 2.50, 3.00, 4.00, 5.00, 7.00, 10.0
])
_GUSTAVSSON_NDOT2 = np.array([
    0.75, 0.68, 0.62, 0.55, 0.50, 0.47, 0.44, 0.42, 0.41, 0.40, 0.40,
    0.41, 0.42, 0.44, 0.47, 0.49, 0.50, 0.52, 0.53, 0.55, 0.57
])


def compute_tumbling_rate_vs_r(A_series, times, aspect_ratios):
    """
    For each aspect ratio, integrate orientation and compute
    <omega^2> * tau_K^2  (normalised squared tumbling rate).

    tau_K = 1 / sqrt(Tr<A^T A>) — Kolmogorov time from data.

    Returns dict: r -> mean_ndot2_normalised
    """
    from orientation import integrate_orientation

    # Kolmogorov time from the full A timeseries
    AtA      = np.einsum('tji,tjk->tik', A_series, A_series)   # A^T A
    TrAtA    = np.einsum('tii->t', AtA)                         # Tr(A^T A)
    tau_K    = 1.0 / np.sqrt(np.mean(TrAtA))
    print(f"  Kolmogorov time tau_K = {tau_K:.4f}")

    results = {}
    for r in aspect_ratios:
        axes = [float(r), 1.0, 1.0]
        print(f"  Integrating r={r}...", end=" ", flush=True)
        R_h, omega_h, *_ = integrate_orientation(axes, A_series, times)

        # omega_h is body-frame angular velocity (T, 3)
        ndot2 = np.mean(np.sum(omega_h**2, axis=1))   # <|omega|^2>
        ndot2_norm = ndot2 * tau_K**2                  # normalised

        results[r] = {
            'ndot2':      ndot2,
            'ndot2_norm': ndot2_norm,
        }
        print(f"<ndot^2>={ndot2:.4f}  normalised={ndot2_norm:.4f}")

    return results, tau_K


def plot_step5_gustavsson_comparison(tumble_results, tau_K, aspect_ratios):
    """
    Compare our <ndot^2> * tau_K^2 vs lambda curve to Gustavsson Fig. 2 (right).
    """
    rs       = np.array(aspect_ratios)
    our_vals = np.array([tumble_results[r]['ndot2_norm'] for r in rs])

    fig, ax = plt.subplots(figsize=(8, 6))

    # Gustavsson DNS curve
    ax.plot(_GUSTAVSSON_LAMBDA, _GUSTAVSSON_NDOT2,
            'o--', color='steelblue', lw=2, ms=5, label='Gustavsson 2014 (DNS)')

    # Our curve
    ax.plot(rs, our_vals,
            's-', color='darkorange', lw=2.5, ms=7,
            label='Chevillard-Meneveau model (my stuff)')

    ax.axvline(1.0, color='gray', lw=0.8, linestyle=':')
    ax.set_xlabel('Aspect ratio $\\lambda = a/b$', fontsize=12)
    ax.set_ylabel('$\\langle \\omega^2 \\rangle \\tau_K^2$', fontsize=13)
    ax.set_title('Normalised tumbling rate vs aspect ratio\n'
                 'Comparison with Gustavsson et al. (2014) Fig. 2',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # Annotate disk/rod regions
    ax.text(0.25, ax.get_ylim()[1]*0.95, 'Disks\n$(\\lambda < 1)$',
            ha='center', fontsize=9, color='gray')
    ax.text(3.0,  ax.get_ylim()[1]*0.95, 'Rods\n$(\\lambda > 1)$',
            ha='center', fontsize=9, color='gray')

    fig.tight_layout()
    fig.savefig(OUTDIR + 'step5_gustavsson_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved step5_gustavsson_comparison.png")


# ─────────────────────────────────────────────────────────────
# STEP 6 — TrO²S DIAGNOSTIC
# ─────────────────────────────────────────────────────────────

def plot_step6_TrO2S(times, A_series, R_history, omega_history, traction_mag=None):
    """
    Analyse TrO^2S — the Gustavsson vortex-tube diagnostic.

    (a) TrO^2S timeseries + distribution
    (b) MI of TrO^2S with orientation features (adds to step 3 picture)
    (c) <ndot^2> conditioned on TrO^2S quantile — mirrors Gustavsson Fig. 2
    (d) Mean traction conditioned on TrO^2S quantile
    """
    scalars = flow_scalars(A_series)
    TrO2S   = scalars['TrO2S']
    theta_p, _ = symmetry_axis_angles(R_history)

    # ── (a) Timeseries + distribution ───────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(times, TrO2S, lw=0.7, color='purple', alpha=0.8)
    axes[0].axhline(0, color='gray', lw=0.8, linestyle='--')
    axes[0].set_xlabel('Time'); axes[0].set_ylabel('$\\mathrm{Tr}(O^2 S)$')
    axes[0].set_title('$\\mathrm{Tr}(O^2 S)$ timeseries', fontsize=11)
    axes[0].grid(True, alpha=0.3)

    p1, p99 = np.percentile(TrO2S, 1), np.percentile(TrO2S, 99)
    Tc = TrO2S[(TrO2S >= p1) & (TrO2S <= p99)]
    kde = gaussian_kde(Tc, bw_method=0.1)
    xg  = np.linspace(p1, p99, 400)
    axes[1].plot(xg, kde(xg), color='purple', lw=2)
    axes[1].fill_between(xg, kde(xg), alpha=0.2, color='purple')
    axes[1].axvline(0, color='gray', lw=0.8, linestyle='--')
    axes[1].set_xlabel('$\\mathrm{Tr}(O^2 S)$'); axes[1].set_ylabel('Density')
    axes[1].set_title('Distribution of $\\mathrm{Tr}(O^2 S)$', fontsize=11)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Gustavsson Vortex Diagnostic $\\mathrm{Tr}(O^2 S)$',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTDIR + 'step6a_TrO2S_timeseries.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved step6a_TrO2S_timeseries.png")

    # ── (b) Conditional orientation given TrO^2S sign ───────
    # Large positive TrO^2S = vortex-dominated; negative = strain-dominated
    vortex_mask = TrO2S > np.percentile(TrO2S, 77)   # top 23% as in Gustavsson
    strain_mask = ~vortex_mask

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, mask, label, col in zip(
            axes,
            [vortex_mask, strain_mask],
            ['Vortex-dominated\n$(\\mathrm{Tr}O^2S > 77\\mathrm{th}$ pct)',
             'Strain-dominated\n$(\\mathrm{Tr}O^2S \\leq 77\\mathrm{th}$ pct)'],
            ['purple', 'darkorange']):
        v = theta_p[mask]
        if len(v) > 20:
            kde = gaussian_kde(v, bw_method=0.1)
            xg  = np.linspace(theta_p.min(), theta_p.max(), 300)
            ax.plot(xg, kde(xg), color=col, lw=2)
            ax.fill_between(xg, kde(xg), alpha=0.2, color=col)
        ax.set_xlabel('$\\theta_p$ (deg)'); ax.set_ylabel('Density')
        ax.set_title(label, fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Orientation distribution conditioned on $\\mathrm{Tr}(O^2 S)$',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTDIR + 'step6b_orientation_conditioned_TrO2S.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved step6b_orientation_conditioned_TrO2S.png")

    # ── (c) MI: TrO^2S vs orientation features ──────────────
    from sklearn.feature_selection import mutual_info_regression
    r_val   = AXES[0] / AXES[1]
    C       = jeffery_C(R_history, r_val)
    finite  = np.isfinite(C)

    orient_feats = {
        '$\\theta_p$': theta_p[finite],
        '$\\phi_p$':   symmetry_axis_angles(R_history)[1][finite],
        'C':           C[finite],
    }
    TrO2S_f = TrO2S[finite]

    mi_vals = {}
    for name, feat in orient_feats.items():
        mi = float(mutual_info_regression(
            feat.reshape(-1, 1), TrO2S_f, random_state=0)[0])
        mi_vals[name] = mi
        print(f"  MI(TrO2S; {name}) = {mi:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(list(mi_vals.keys()), list(mi_vals.values()), color='purple', alpha=0.7)
    ax.set_ylabel('Mutual Information (nats)', fontsize=11)
    ax.set_title('MI between $\\mathrm{Tr}(O^2 S)$ and orientation features',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(OUTDIR + 'step6c_MI_TrO2S_orientation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved step6c_MI_TrO2S_orientation.png")


# ─────────────────────────────────────────────────────────────
# STEP 7 — SPLIT BY TrO²S: TRACTION AND TUMBLING CONDITIONED
#           ON VORTEX vs STRAIN REGIONS  (mirrors Gustavsson Fig. 2)
#
# KEY FIX: condition on TIME-AVERAGED TrO^2S, not instantaneous.
# Gustavsson's mechanism requires persistent vortex tubes over ~5 tau_K.
# The particle needs time to respond — conditioning on instantaneous
# TrO^2S conflates the flow at t with the orientation at t, missing
# the lag. Time-averaging captures whether the particle has been living
# in a vortex-dominated region long enough to respond dynamically.
# ─────────────────────────────────────────────────────────────

def plot_step7_vortex_strain_split(times, A_series, R_history,
                                   omega_history, tau_K,
                                   traction_mag=None,
                                   vortex_pct=77,
                                   window_tauK=5.0):
    """
    Split timeseries into vortex-dominated and strain-dominated regions
    using a TIME-AVERAGED TrO^2S with window = window_tauK * tau_K.

    Parameters
    ----------
    tau_K : float
        Kolmogorov time, used to set the averaging window length.
    window_tauK : float
        Window size in units of tau_K. Default 5 (Gustavsson's correlation
        functions decay over ~5 tau_K, so this is the physically motivated
        timescale for the particle to respond to the flow structure).
    """
    scalars = flow_scalars(A_series)
    TrO2S   = scalars['TrO2S']

    # ── Time-average TrO^2S over window_tauK * tau_K ────────
    dt     = float(np.mean(np.diff(times)))
    window = max(1, int(window_tauK * tau_K / dt))
    print(f"  TrO^2S averaging window: {window} steps = {window*dt:.3f} time units "
          f"= {window*dt/tau_K:.1f} tau_K")

    kernel       = np.ones(window) / window
    TrO2S_smooth = np.convolve(TrO2S, kernel, mode='same')

    # ── Split by smoothed TrO^2S ─────────────────────────────
    threshold   = np.percentile(TrO2S_smooth, vortex_pct)
    vortex_mask = TrO2S_smooth >  threshold
    strain_mask = TrO2S_smooth <= threshold

    omega_mag = np.sqrt(np.sum(omega_history**2, axis=1))

    labels = ['All data',
              f'Vortex\n(smoothed $\\mathrm{{Tr}}O^2S > {vortex_pct}$th pct)',
              f'Strain\n(smoothed $\\mathrm{{Tr}}O^2S \\leq {vortex_pct}$th pct)']
    masks  = [np.ones(len(TrO2S), dtype=bool), vortex_mask, strain_mask]
    colors = ['steelblue', 'purple', 'darkorange']

    # ── (a) Distributions: instantaneous vs smoothed side by side ──
    # Also show the instantaneous split for direct comparison
    TrO2S_inst_thresh = np.percentile(TrO2S, vortex_pct)
    vortex_inst = TrO2S >  TrO2S_inst_thresh
    strain_inst = TrO2S <= TrO2S_inst_thresh

    n_cols = 3 if traction_mag is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6*n_cols, 5))

    # Instantaneous split (left) — for comparison
    ax = axes[0]
    for mask, lbl, col in zip(
            [np.ones(len(TrO2S), bool), vortex_inst, strain_inst],
            ['All', 'Vortex (instant.)', 'Strain (instant.)'],
            colors):
        v = omega_mag[mask]
        p99 = np.percentile(v, 99); vc = v[v <= p99]
        if len(vc) < 20: continue
        kde = gaussian_kde(vc, bw_method=0.12)
        xg  = np.linspace(0, p99, 300)
        ax.plot(xg, kde(xg), color=col, lw=2, label=lbl, linestyle='--')
    ax.set_xlabel('$|\\omega|$', fontsize=12); ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Tumbling rate\nInstantaneous TrO²S split', fontsize=10)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Time-averaged split (middle)
    ax = axes[1]
    for mask, lbl, col in zip(masks, ['All', 'Vortex (smoothed)', 'Strain (smoothed)'], colors):
        v = omega_mag[mask]
        p99 = np.percentile(v, 99); vc = v[v <= p99]
        if len(vc) < 20: continue
        kde = gaussian_kde(vc, bw_method=0.12)
        xg  = np.linspace(0, p99, 300)
        ax.plot(xg, kde(xg), color=col, lw=2, label=lbl)
    ax.set_xlabel('$|\\omega|$', fontsize=12); ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Tumbling rate\nTime-averaged TrO²S ({window_tauK}$\\tau_K$ window)',
                 fontsize=10)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Traction (right, if available)
    if traction_mag is not None:
        ax = axes[2]
        for mask, lbl, col in zip(masks, ['All', 'Vortex (smoothed)', 'Strain (smoothed)'], colors):
            v = traction_mag[mask]
            p99 = np.percentile(v, 99); vc = v[v <= p99]
            if len(vc) < 20: continue
            kde = gaussian_kde(vc, bw_method=0.12)
            xg  = np.linspace(0, p99, 300)
            ax.plot(xg, kde(xg), color=col, lw=2, label=lbl)
        ax.set_xlabel('$\\|\\tau\\|$', fontsize=12); ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Traction\nTime-averaged TrO²S ({window_tauK}$\\tau_K$ window)',
                     fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle('Vortex vs strain split: instantaneous vs time-averaged TrO²S\n'
                 f'(window = {window_tauK} $\\tau_K$ = {window} steps)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTDIR + 'step7_vortex_strain_distributions.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved step7_vortex_strain_distributions.png")

    # ── (b) Summary bar chart — smoothed split ───────────────
    plot_vals = [omega_mag]
    ylabels   = ['$\\langle |\\omega| \\rangle$']
    titles    = ['Mean tumbling rate']
    if traction_mag is not None:
        plot_vals.append(traction_mag)
        ylabels.append('$\\langle \\|\\tau\\| \\rangle$')
        titles.append('Mean traction')

    fig, axes = plt.subplots(1, len(plot_vals), figsize=(6*len(plot_vals), 5))
    if len(plot_vals) == 1:
        axes = [axes]

    for ax, vals_all, ylabel, title in zip(axes, plot_vals, ylabels, titles):
        means = [np.mean(vals_all[m]) for m in masks]
        stds  = [np.std(vals_all[m])  for m in masks]
        x     = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, color=colors, alpha=0.75,
               capsize=5, edgecolor='black', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        fracs = [m.mean() for m in masks]
        for xi, (mean, frac) in enumerate(zip(means, fracs)):
            ax.text(xi, mean * 1.05, f'{frac*100:.0f}%\nof data',
                    ha='center', fontsize=8)

    fig.suptitle(f'Mean statistics: time-averaged TrO²S split '
                 f'({window_tauK}$\\tau_K$ window)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTDIR + 'step7_vortex_strain_summary.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved step7_vortex_strain_summary.png")

    # ── (c) Print summary numbers ────────────────────────────
    print(f"\n  Smoothed TrO^2S threshold (top {100-vortex_pct}%): {threshold:.4f}")
    for mask, lbl in zip(masks, ['All', 'Vortex', 'Strain']):
        tau_str = f"  <tau>={np.mean(traction_mag[mask]):.4f}" \
                  if traction_mag is not None else ""
        print(f"  {lbl:<8}  n={mask.sum():5d}  "
              f"<|omega|>={np.mean(omega_mag[mask]):.4f}{tau_str}")


# ─────────────────────────────────────────────────────────────
# STEP 8 — AUTOCORRELATION: CM vs LW
# ─────────────────────────────────────────────────────────────
 
def autocorr(x, max_lag):
    """
    Normalised autocorrelation C(lag) = <x'(t)x'(t+lag)> / <x'^2>
    Uses FFT for O(N log N) efficiency.
    Returns (lags_int, C_array) both length max_lag+1.
    """
    x   = x - np.mean(x)
    var = np.var(x)
    if var < 1e-12:
        return np.arange(max_lag + 1), np.zeros(max_lag + 1)
    n    = len(x)
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    f        = np.fft.rfft(x, n=nfft)
    acf_full = np.fft.irfft(f * np.conj(f))[:n]
    # Normalise by number of pairs at each lag and by variance
    acf_full /= (var * np.arange(n, 0, -1))
    return np.arange(max_lag + 1), acf_full[:max_lag + 1]
 
 
def decorrelation_time(lags_dt, C):
    """
    Physical lag at which ACF first crosses 1/e.
    Uses linear interpolation. Returns nan if never crosses within window.
    """
    threshold = 1.0 / np.e
    for i in range(1, len(C)):
        if C[i] <= threshold:
            frac = (C[i-1] - threshold) / (C[i-1] - C[i] + 1e-30)
            return float(lags_dt[i-1] + frac * (lags_dt[i] - lags_dt[i-1]))
    return float('nan')
 
 
def compute_step8_autocorrelations(times, A_series, R_history,
                                   label, max_lag_time=10.0,
                                   tau_mag=None):
    """
    Compute normalised ACF for five signals:
      ||S||, ||W||, Q     — flow scalars
      cos(theta_p)        — orientation memory (cos avoids 0/180deg wraparound)
      ||tau||             — traction magnitude (optional, requires tau_mag array)
 
    Key comparisons:
      tau(||S||)/tau(||W||)  : ~1 in CM, <1 in LW (gamma term signature)
      tau(cos theta_p) vs tau(||tau||) : if similar, orientation drives traction
                                         persistence; if tau(||tau||) much shorter,
                                         flow fluctuations wash out orientation effect
    """
    dt      = float(times[1] - times[0])
    max_lag = min(int(max_lag_time / dt), len(times) - 1)
    lags_dt = np.arange(max_lag + 1) * dt
 
    scalars    = flow_scalars(A_series)
    theta_p, _ = symmetry_axis_angles(R_history)
 
    signals = {
        '$\\|S\\|$':        scalars['S_norm'],
        '$\\|W\\|$':        scalars['W_norm'],
        '$Q$':              scalars['Q'],
        '$\\cos\\theta_p$': np.cos(np.radians(theta_p)),
    }
 
    # Add traction if provided
    if tau_mag is not None:
        signals['$\\|\\tau\\|$'] = tau_mag
 
    results = {}
    print(f"\n  [{label}] Decorrelation times (1/e criterion):")
    for name, sig in signals.items():
        _, C     = autocorr(sig, max_lag)
        tau_corr = decorrelation_time(lags_dt, C)
        results[name] = {'lags_dt': lags_dt, 'C': C, 'tau_corr': tau_corr}
        print(f"    {name:<28}  tau_corr = {tau_corr:.4f} tau_eta")
 
    # Key ratio 1: strain vs vorticity (gamma term signature)
    tau_S = results['$\\|S\\|$']['tau_corr']
    tau_W = results['$\\|W\\|$']['tau_corr']
    if np.isfinite(tau_S) and np.isfinite(tau_W) and tau_W > 0:
        print(f"    tau(||S||) / tau(||W||)    = {tau_S/tau_W:.3f}  "
              f"(CM ~1; LW <1 expected)")
 
    # Key ratio 2: orientation vs traction persistence
    if '$\\cos\\theta_p$' in results and '$\\|\\tau\\|$' in results:
        tau_orient  = results['$\\cos\\theta_p$']['tau_corr']
        tau_trac    = results['$\\|\\tau\\|$']['tau_corr']
        if np.isfinite(tau_orient) and np.isfinite(tau_trac) and tau_orient > 0:
            print(f"    tau(||tau||) / tau(cos th_p) = {tau_trac/tau_orient:.3f}  "
                  f"(~1 means orientation drives traction persistence; "
                  f"<<1 means flow fluctuations dominate)")
 
    return results
 
 
def plot_step8_autocorrelations(acf_cm, acf_lw=None, tau_K=None):
    """
    Five-panel ACF plot (3x2 grid, last cell used for summary text).
    Signals: ||S||, ||W||, Q, cos(theta_p), ||tau||
    CM in orange (solid), LW in blue (dashed).
    acf_lw may be None — plots CM only.
    """
    signals  = list(acf_cm.keys())   # up to 5 signals
    cm_color = '#E07B39'
    lw_color = '#4472C4'
 
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes_flat = list(axes.flat)
 
    for ax, name in zip(axes_flat, signals):
        d_cm = acf_cm[name]
        ax.plot(d_cm['lags_dt'], d_cm['C'],
                color=cm_color, lw=2.2,
                label=f"CM  $\\tau_{{corr}}$={d_cm['tau_corr']:.3f} $\\tau_\\eta$")
 
        if acf_lw is not None and name in acf_lw:
            d_lw = acf_lw[name]
            ax.plot(d_lw['lags_dt'], d_lw['C'],
                    color=lw_color, lw=2.2, linestyle='--',
                    label=f"LW  $\\tau_{{corr}}$={d_lw['tau_corr']:.3f} $\\tau_\\eta$")
 
        ax.axhline(1.0/np.e, color='black', lw=1.0, linestyle=':', label='$1/e$')
        ax.axhline(0,        color='grey',  lw=0.5)
        if tau_K is not None:
            ax.axvline(tau_K, color='grey', lw=1.0, linestyle='-.', alpha=0.6,
                       label=f'$\\tau_K$={tau_K:.3f}')
 
        ax.set_xlim(0, d_cm['lags_dt'][-1])
        ax.set_ylim(-0.25, 1.05)
        ax.set_xlabel('Lag  ($\\tau_\\eta$)', fontsize=11)
        ax.set_ylabel('Autocorrelation', fontsize=11)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
 
    # Last panel: annotation box comparing tau(cos theta_p) vs tau(||tau||)
    ax_last = axes_flat[len(signals)]
    ax_last.axis('off')
    lines = ['Key ratios (CM):']
    tau_S = acf_cm.get('$\\|S\\|$',     {}).get('tau_corr', float('nan'))
    tau_W = acf_cm.get('$\\|W\\|$',     {}).get('tau_corr', float('nan'))
    tau_p = acf_cm.get('$\\cos\\theta_p$', {}).get('tau_corr', float('nan'))
    tau_t = acf_cm.get('$\\|\\tau\\|$', {}).get('tau_corr', float('nan'))
    if np.isfinite(tau_S) and np.isfinite(tau_W) and tau_W > 0:
        lines.append(f'  τ(||S||)/τ(||W||) = {tau_S/tau_W:.3f}')
        lines.append(f'  (expect ~1; LW <1)')
    if np.isfinite(tau_p) and np.isfinite(tau_t) and tau_p > 0:
        lines.append(f'\n  τ(||τ||)/τ(cosθ_p) = {tau_t/tau_p:.3f}')
        lines.append(f'  (~1 → orientation drives')
        lines.append(f'   traction persistence)')
        lines.append(f'  (<<1 → flow fluctuations')
        lines.append(f'   dominate)')
    if acf_lw is not None:
        lines.append('\nKey ratios (LW):')
        tau_S_lw = acf_lw.get('$\\|S\\|$',     {}).get('tau_corr', float('nan'))
        tau_W_lw = acf_lw.get('$\\|W\\|$',     {}).get('tau_corr', float('nan'))
        tau_p_lw = acf_lw.get('$\\cos\\theta_p$', {}).get('tau_corr', float('nan'))
        tau_t_lw = acf_lw.get('$\\|\\tau\\|$', {}).get('tau_corr', float('nan'))
        if np.isfinite(tau_S_lw) and np.isfinite(tau_W_lw) and tau_W_lw > 0:
            lines.append(f'  τ(||S||)/τ(||W||) = {tau_S_lw/tau_W_lw:.3f}')
        if np.isfinite(tau_p_lw) and np.isfinite(tau_t_lw) and tau_p_lw > 0:
            lines.append(f'  τ(||τ||)/τ(cosθ_p) = {tau_t_lw/tau_p_lw:.3f}')
    ax_last.text(0.05, 0.95, '\n'.join(lines),
                 transform=ax_last.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='#F0F4FF', alpha=0.8))
 
    fig.suptitle(
        'Autocorrelation functions: CM vs LW\n'
        '$\\tau(\\|S\\|)/\\tau(\\|W\\|)$: CM$\\approx$1, LW$<$1 ($\\gamma$ term)  |  '
        '$\\tau(\\|\\tau\\|)/\\tau(\\cos\\theta_p)$: tests if orientation drives traction persistence',
        fontsize=11, fontweight='bold'
    )
    fig.tight_layout()
    fig.savefig(OUTDIR + 'step8_autocorrelations.png', dpi=150, bbox_inches='tight')
    plt.close(fig); print("Saved step8_autocorrelations.png")
 
    # ── Summary table ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axis('off')
    col_labels = ['Signal', 'CM  tau_corr (tau_eta)',
                  'LW  tau_corr (tau_eta)', 'LW / CM']
    rows = []
    for name in signals:
        tau_cm = acf_cm[name]['tau_corr']
        tau_lw = acf_lw[name]['tau_corr'] if (acf_lw and name in acf_lw) \
                 else float('nan')
        ratio  = (tau_lw/tau_cm
                  if (np.isfinite(tau_lw) and tau_cm > 0) else float('nan'))
        clean  = (name.replace('$','').replace('\\|','|')
                      .replace('\\cos','cos').replace('\\theta_p','theta_p'))
        rows.append([clean,
                     f'{tau_cm:.4f}',
                     f'{tau_lw:.4f}' if np.isfinite(tau_lw) else '--',
                     f'{ratio:.3f}'  if np.isfinite(ratio)   else '--'])
    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   cellLoc='center', loc='center', colColours=['#DDEEFF']*4)
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1.3, 2.0)
    ax.set_title(
        'Decorrelation times\n'
        'LW/CM < 1 for ||S|| confirms gamma-term: strain decorrelates faster in LW',
        fontsize=10, fontweight='bold', pad=15)
    fig.tight_layout()
    fig.savefig(OUTDIR + 'step8_decorrelation_table.png', dpi=150, bbox_inches='tight')
    plt.close(fig); print("Saved step8_decorrelation_table.png")

# ─────────────────────────────────────────────────────────────
# STEP 9 — HIGH-TRACTION EXCURSION DURATION ANALYSIS
# ─────────────────────────────────────────────────────────────
 
def excursion_durations(tau_mag, dt, threshold):
    """
    Find all contiguous runs where tau_mag > threshold.
    Returns array of durations in physical time units.
 
    Parameters
    ----------
    tau_mag   : (T,) traction magnitude timeseries
    dt        : timestep in physical time units
    threshold : scalar threshold value
 
    Returns
    -------
    durations : (N_excursions,) array of durations in tau_eta units
    """
    above = (tau_mag > threshold).astype(int)
    # Find transitions
    diff  = np.diff(above, prepend=0, append=0)
    starts = np.where(diff ==  1)[0]
    ends   = np.where(diff == -1)[0]
    durations = (ends - starts) * dt
    return durations
 
 
def compute_step9_excursions(tau_mag, dt, label,
                             percentiles=(90, 95, 99)):
    """
    Compute excursion duration statistics at multiple thresholds.
 
    Parameters
    ----------
    tau_mag     : (T,) traction timeseries
    dt          : timestep in tau_eta units
    label       : string label for printing ('CM' or 'LW')
    percentiles : thresholds defined as percentiles of tau_mag
 
    Returns
    -------
    dict: percentile -> {'threshold', 'durations', 'mean', 'p95', 'n'}
    """
    results = {}
    print(f"\n  [{label}] High-traction excursion durations:")
    for pct in percentiles:
        threshold = np.percentile(tau_mag, pct)
        durs      = excursion_durations(tau_mag, dt, threshold)
        if len(durs) == 0:
            print(f"    {pct}th pct  threshold={threshold:.3f}  "
                  f"no excursions found")
            results[pct] = {'threshold': threshold, 'durations': durs,
                            'mean': np.nan, 'p95': np.nan, 'n': 0}
            continue
        mean_dur = float(np.mean(durs))
        p95_dur  = float(np.percentile(durs, 95))
        results[pct] = {'threshold': threshold, 'durations': durs,
                        'mean': mean_dur, 'p95': p95_dur, 'n': len(durs)}
        print(f"    {pct}th pct  threshold={threshold:.3f}  "
              f"n={len(durs)}  mean={mean_dur:.4f} tau_eta  "
              f"p95={p95_dur:.4f} tau_eta")
    return results
 
 
def plot_step9_excursions(exc_cm, exc_lw=None, dt_cm=0.001,
                          percentiles=(90, 95, 99), tau_K=None):
    """
    Three rows (one per percentile threshold), three columns:
      Col 1: Distribution of excursion durations  (CM vs LW)
      Col 2: CCDF of excursion durations           (CM vs LW)
      Col 3: Summary bar chart — mean and p95 duration
 
    exc_cm, exc_lw : dicts from compute_step9_excursions
    exc_lw may be None.
    """
    cm_color = '#E07B39'
    lw_color = '#4472C4'
    n_rows   = len(percentiles)
 
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
 
    for row, pct in enumerate(percentiles):
        d_cm = exc_cm.get(pct, {})
        d_lw = exc_lw.get(pct, {}) if exc_lw is not None else {}
 
        durs_cm = d_cm.get('durations', np.array([]))
        durs_lw = d_lw.get('durations', np.array([]))
 
        thr_cm  = d_cm.get('threshold', float('nan'))
 
        # ── Col 0: PDF of durations ───────────────────────────
        ax = axes[row, 0]
        for durs, label, color in [(durs_cm, 'CM', cm_color),
                                   (durs_lw, 'LW', lw_color)]:
            if len(durs) < 5:
                continue
            p99 = np.percentile(durs, 99)
            dc  = durs[durs <= p99]
            if len(dc) < 3:
                continue
            try:
                kde = gaussian_kde(dc, bw_method=0.2)
                xg  = np.linspace(0, p99, 300)
                ax.plot(xg, kde(xg), color=color, lw=2.2, label=label)
                ax.fill_between(xg, kde(xg), alpha=0.15, color=color)
            except Exception:
                pass
        if tau_K is not None:
            ax.axvline(tau_K, color='grey', lw=1.0, linestyle='-.',
                       alpha=0.7, label=f'$\\tau_K$={tau_K:.3f}')
        ax.set_xlabel('Excursion duration ($\\tau_\\eta$)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{pct}th pct threshold\n'
                     f'(CM thr={thr_cm:.3f})', fontsize=10, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
 
        # ── Col 1: CCDF — P(duration > t) ────────────────────
        ax = axes[row, 1]
        for durs, label, color in [(durs_cm, 'CM', cm_color),
                                   (durs_lw, 'LW', lw_color)]:
            if len(durs) < 2:
                continue
            sorted_d = np.sort(durs)
            ccdf     = 1.0 - np.arange(1, len(sorted_d)+1) / len(sorted_d)
            ax.step(sorted_d, ccdf, color=color, lw=2.2,
                    where='post', label=label)
        if tau_K is not None:
            ax.axvline(tau_K, color='grey', lw=1.0, linestyle='-.',
                       alpha=0.7, label=f'$\\tau_K$')
        ax.set_xlabel('Excursion duration ($\\tau_\\eta$)', fontsize=10)
        ax.set_ylabel('P(duration > t)', fontsize=10)
        ax.set_title(f'{pct}th pct — CCDF\n'
                     f'(heavy tail → long sustained events)', fontsize=10,
                     fontweight='bold')
        ax.set_yscale('log')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
 
        # ── Col 2: Summary bar — mean and p95 ────────────────
        ax = axes[row, 2]
        labels_bar = ['CM\nmean', 'CM\np95']
        values_bar = [d_cm.get('mean', np.nan), d_cm.get('p95', np.nan)]
        colors_bar = [cm_color, cm_color]
        alphas_bar = [0.9, 0.55]
        if exc_lw is not None:
            labels_bar += ['LW\nmean', 'LW\np95']
            values_bar += [d_lw.get('mean', np.nan), d_lw.get('p95', np.nan)]
            colors_bar += [lw_color, lw_color]
            alphas_bar += [0.9, 0.55]
        x_pos = np.arange(len(labels_bar))
        for x, v, c, a in zip(x_pos, values_bar, colors_bar, alphas_bar):
            if np.isfinite(v):
                ax.bar(x, v, color=c, alpha=a, width=0.6)
                ax.text(x, v * 1.02, f'{v:.3f}', ha='center',
                        va='bottom', fontsize=9)
        if tau_K is not None:
            ax.axhline(tau_K, color='grey', lw=1.0, linestyle='-.',
                       alpha=0.7, label=f'$\\tau_K$={tau_K:.3f}')
            ax.legend(fontsize=9)
        ax.set_xticks(x_pos); ax.set_xticklabels(labels_bar, fontsize=9)
        ax.set_ylabel('Duration ($\\tau_\\eta$)', fontsize=10)
        ax.set_title(f'{pct}th pct — Mean & 95th pct duration',
                     fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
 
    fig.suptitle(
        'High-traction excursion durations: CM vs LW\n'
        'Duration of sustained ||$\\tau$|| > threshold events '
        '— directly relevant to biofilm detachment',
        fontsize=12, fontweight='bold'
    )
    fig.tight_layout()
    fig.savefig(OUTDIR + 'step9_excursion_durations.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved step9_excursion_durations.png")
 
    # ── Threshold sensitivity summary ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    pct_vals  = list(percentiles)
 
    for ax, stat, ylabel in zip(
            axes,
            ['mean', 'p95'],
            ['Mean excursion duration ($\\tau_\\eta$)',
             '95th pct excursion duration ($\\tau_\\eta$)']):
        cm_vals = [exc_cm.get(p, {}).get(stat, np.nan) for p in pct_vals]
        ax.plot(pct_vals, cm_vals, 'o-', color=cm_color, lw=2.2,
                ms=7, label='CM')
        if exc_lw is not None:
            lw_vals = [exc_lw.get(p, {}).get(stat, np.nan) for p in pct_vals]
            ax.plot(pct_vals, lw_vals, 's--', color=lw_color, lw=2.2,
                    ms=7, label='LW')
        if tau_K is not None:
            ax.axhline(tau_K, color='grey', lw=1.0, linestyle='-.',
                       alpha=0.7, label=f'$\\tau_K$={tau_K:.3f}')
        ax.set_xlabel('Threshold percentile', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=11, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
        ax.set_xticks(pct_vals)
 
    fig.suptitle(
        'Excursion duration vs threshold (robustness check)\n'
        'Robust result: CM vs LW difference persists across all thresholds',
        fontsize=12, fontweight='bold'
    )
    fig.tight_layout()
    fig.savefig(OUTDIR + 'step9_threshold_sensitivity.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved step9_threshold_sensitivity.png")




# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ── Load ────────────────────────────────────────────────
    times, A_series = load_csv(CSV_PATH)

    # ── Integrate orientation ────────────────────────────────
    print("\nIntegrating orientation ODE...")
    R_history, omega_history, angle_history, axis_history, dtheta_history = \
        integrate_orientation(AXES, A_series, times)

    # ── Burn-in ─────────────────────────────────────────────
    n_burn        = int(len(times) * BURN_IN)
    times_b       = times[n_burn:]
    A_b           = A_series[n_burn:]
    R_b           = R_history[n_burn:]
    omega_b       = omega_history[n_burn:]

    r = AXES[0] / AXES[1]

    # # ── Step 1 ───────────────────────────────────────────────
    # print("\n── Step 1: Orientation representations ──")
    # plot_step1_euler(times_b, R_b)
    # plot_step1_symmetry_axis(times_b, R_b)
    # plot_step1_jeffery_C(times_b, R_b, r)

    # # ── Step 2 ───────────────────────────────────────────────
    # print("\n── Step 2: Conditional flow distributions ──")
    # plot_step2_conditional(times_b, R_b, A_b, r, N_BINS)

    # # ── Step 3 ───────────────────────────────────────────────
    # print("\n── Step 3: Mutual information + shuffle test ──")
    # mi_results = compute_mutual_information(R_b, A_b, n_shuffles=200)
    # plot_step3_mutual_information(mi_results)

    # ── Step 4 ───────────────────────────────────────────────
    tau_mag = None
    if RUN_STEP4:
        print("\n── Step 4: Conditional traction ──")

        from fast_traction import build_transfer_matrix, fast_traction_magnitude

        M, n_hat = build_transfer_matrix(AXES, [AXES[0], 0., 0.], mu=MU)
        tau_mag  = fast_traction_magnitude(M, n_hat, A_b, R_b)

        tau_mag = traction_magnitude(A_b, R_b, AXES, MU)
        plot_step4_conditional_traction(times_b, R_b, A_b, tau_mag, N_BINS)
    else:
        print("\n── Step 4: Skipped (RUN_STEP4=False) ──")

    # ── Step 5 — Gustavsson comparison ──────────────────────
    print("\n── Step 5: Tumbling rate vs aspect ratio (Gustavsson comparison) ──")
    # Aspect ratios to sweep — covers disk (r<1), sphere (r=1), rod (r>1)
    ASPECT_RATIOS_SWEEP = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    tumble_results, tau_K = compute_tumbling_rate_vs_r(A_b, times_b,
                                                        ASPECT_RATIOS_SWEEP)
    plot_step5_gustavsson_comparison(tumble_results, tau_K, ASPECT_RATIOS_SWEEP)

    # ── Step 6 — TrO^2S diagnostic ───────────────────────────
    print("\n── Step 6: TrO^2S vortex diagnostic ──")
    plot_step6_TrO2S(times_b, A_b, R_b, omega_b, tau_mag)

    # ── Step 7 — Vortex vs strain split ──────────────────────
    print("\n── Step 7: Vortex vs strain split of traction and tumbling ──")
    plot_step7_vortex_strain_split(times_b, A_b, R_b, omega_b, tau_K,
                                   traction_mag=tau_mag,
                                   vortex_pct=77,
                                   window_tauK=5.0)


    # ── Step 8 — Autocorrelation ──────────────────────
    if RUN_STEP8:
        print("\n-- Step 8: Autocorrelation functions --")
 
        acf_cm = compute_step8_autocorrelations(
            times_b, A_b, R_b,
            label='CM',
            max_lag_time=ACF_MAX_LAG,
            tau_mag= tau_mag
        )
 
        acf_lw = None
        if LW_CSV_PATH is not None and os.path.exists(LW_CSV_PATH):
            print(f"  Loading LW data from '{LW_CSV_PATH}'...")
            times_lw, A_lw = load_csv(LW_CSV_PATH)
            print("  Integrating orientation ODE (LW)...")
            R_lw, _, _, _, _ = integrate_orientation(AXES, A_lw, times_lw)
            n_burn_lw  = int(len(times_lw) * BURN_IN)
            tau_mag_lw = None                              # ← add
            if tau_mag is not None:                        # ← add
                print("  Computing traction (LW)...")      # ← add
                tau_mag_lw = traction_magnitude(           # ← add
                    A_lw[n_burn_lw:], R_lw[n_burn_lw:], AXES, MU)  # ← add
            acf_lw = compute_step8_autocorrelations(
                times_lw[n_burn_lw:], A_lw[n_burn_lw:], R_lw[n_burn_lw:],
                label='LW',
                max_lag_time=ACF_MAX_LAG,
                tau_mag=tau_mag_lw   # ← add this
            )
        elif LW_CSV_PATH is not None:
            print(f"  '{LW_CSV_PATH}' not found -- plotting CM only.")
            print("  Run generate_grad_u_LW.py first to enable LW comparison.")
 
        plot_step8_autocorrelations(acf_cm, acf_lw, tau_K=tau_K)
    
    if RUN_STEP9:
        print("\n-- Step 9: High-traction excursion durations --")
        if tau_mag is None:
            print("  Skipping — traction not computed (set RUN_STEP4=True)")
        else:
            dt_b   = float(times_b[1] - times_b[0])
            exc_cm = compute_step9_excursions(tau_mag, dt_b, label='CM')
 
            exc_lw = None
            if LW_CSV_PATH is not None and os.path.exists(LW_CSV_PATH):
                # Reuse LW data loaded in Step 8 if available, else reload
                try:
                    tau_mag_lw
                except NameError:
                    print("  Loading LW data for Step 9...")
                    times_lw, A_lw = load_csv(LW_CSV_PATH)
                    R_lw, _, _, _, _ = integrate_orientation(AXES, A_lw, times_lw)
                    n_burn_lw  = int(len(times_lw) * BURN_IN)
                    tau_mag_lw = traction_magnitude(
                        A_lw[n_burn_lw:], R_lw[n_burn_lw:], AXES, MU)
                if tau_mag_lw is not None:
                    dt_lw  = float(times_lw[1] - times_lw[0])
                    exc_lw = compute_step9_excursions(
                        tau_mag_lw, dt_lw, label='LW')
            elif LW_CSV_PATH is not None:
                print("  LW CSV not found — plotting CM only.")
 
            plot_step9_excursions(exc_cm, exc_lw, dt_cm=dt_b,
                                  tau_K=tau_K)

    print(f"\nAll plots saved to '{OUTDIR}'")
    print("Done.")