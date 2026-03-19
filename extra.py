"""
mfpt_surface_and_pdf_mapping.py
================================
Two analyses:

PART A — MFPT colourmap over the full ellipsoid surface
---------------------------------------------------------
  For a grid of (theta, phi) surface points:
    1. Build the 9x9 stress transfer matrix M at each point (once).
    2. Compute ||sigma(t)||_F at every timestep via  vec_sigma = M @ vec(A_body).
    3. Extract recurrence times at each threshold c.
    4. Colour the ellipsoid surface by MFPT(x, c).
  Produces one 3-D colourmap figure per threshold.

PART B — Analytical PDF mapping  A-GMM  ->  ||sigma||_F PDF
-------------------------------------------------------------
  Because vec(sigma) = M @ vec(A) is linear, if vec(A) ~ GMM then
  for each mixture component k:

      vec(sigma) | k  ~  N( M mu_k,  M Sigma_k M^T )   (9-D Gaussian)

  The Frobenius norm squared is a quadratic form of a Gaussian:
      s^2 = ||vec(sigma)||^2 = vec(sigma)^T I vec(sigma)

  For a non-central Gaussian  z ~ N(mu, C), the PDF of ||z|| can be
  evaluated numerically by Monte Carlo over that single Gaussian component
  (cheap — no Ellipsoid calls, just numpy).  The full mixture PDF is:

      p(s) = sum_k  w_k * p_k(s)

  where p_k is the PDF of ||N(M mu_k, M Sigma_k M^T)||.

  This is compared against the empirical KDE from the time series to
  validate the linear-GMM propagation.

  Per-point analysis for the three named surface points, plus a summary.

Inputs
------
  grad_u.csv             time series of A(t)
  gmm_coefficients.csv   fitted GMM for each A_ij component

Outputs
-------
  mfpt_surface/mfpt_surface_c{threshold}.png  — colourmap per threshold
  pdf_mapping/pdf_mapping_point{k}.png         — analytical vs empirical PDF
  pdf_mapping_gmm.csv                          — GMM fit to ||sigma||_F PDF
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
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

CSV_PATH    = "grad_u.csv"
A_GMM_CSV   = "gmm_coefficients.csv"
DIR_SURFACE = "mfpt_surface"
DIR_PDF     = "pdf_mapping"
PDF_GMM_CSV = "pdf_mapping_gmm.csv"

a  = np.array([2.0, 1.0, 1.0])
mu = 1.0

# Named surface points for Part B
named_points = [
    np.array([2.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, 0.0, 1.0]),
]

# Thresholds for Part A colourmap  (one figure per threshold)
# Leave [] to skip Part A and only run Part B
THRESHOLDS = [5, 10, 15, 20]   # e.g. [0.5, 1.0, 1.5, 2.0]

# Surface grid resolution for Part A
N_THETA = 60   # coarser = faster; 40x40 = 1600 points
N_PHI   = 60

# Number of MC draws per GMM component for analytical PDF (Part B)
N_ANALYTIC = 100000

N_GMM_COMP = 3

os.makedirs(DIR_SURFACE, exist_ok=True)
os.makedirs(DIR_PDF,     exist_ok=True)

# ===========================================================================
# SHARED UTILITIES
# ===========================================================================

def ellipsoid_surface_grid(a, n_theta, n_phi):
    """
    Parametric grid on the ellipsoid surface.
    Returns xyz (n_theta*n_phi, 3) and the (theta, phi) angles.
    """
    theta = np.linspace(0, np.pi,   n_theta)   # polar
    phi   = np.linspace(0, 2*np.pi, n_phi)     # azimuthal
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    X = a[0] * np.sin(TH) * np.cos(PH)
    Y = a[1] * np.sin(TH) * np.sin(PH)
    Z = a[2] * np.cos(TH)
    xyz = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)  # (N, 3)
    return xyz, TH, PH


def build_stress_transfer_matrix(ellipsoid_class, a, mu, x_surface):
    """9x9 matrix:  vec(sigma) = M @ vec(A_body)."""
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


# ===========================================================================
# LOAD DATA AND INTEGRATE ORIENTATION (shared by both parts)
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

# Rotate A into body frame once — used by both parts
tmp        = np.einsum('tji,tjk->tik', R_history, A_series)
A_body_ts  = np.einsum('tij,tjk->tik', tmp, R_history)   # (T, 3, 3)
vec_A_body = A_body_ts.reshape(T, 9)                       # (T, 9)

# ===========================================================================
# PART A — MFPT COLOURMAP OVER FULL ELLIPSOID SURFACE
# ===========================================================================

if THRESHOLDS:
    print("\n" + "="*55)
    print("PART A: MFPT colourmap over ellipsoid surface")
    print("="*55)

    # Build surface grid
    xyz, TH, PH = ellipsoid_surface_grid(a, N_THETA, N_PHI)
    N_pts = len(xyz)
    print(f"  Surface grid: {N_THETA}x{N_PHI} = {N_pts} points")

    # Build transfer matrix at every surface point
    print("  Building transfer matrices (this takes a moment)...")
    t0 = time.perf_counter()
    M_grid = np.zeros((N_pts, 9, 9))
    for idx, x_pt in enumerate(xyz):
        if idx % 200 == 0:
            print(f"    {idx}/{N_pts}...", flush=True)
        M_grid[idx] = build_stress_transfer_matrix(Ellipsoid, a, mu, x_pt)
    print(f"  Done in {time.perf_counter()-t0:.1f}s")

    # Frobenius norm time series at all surface points
    # vec_sigma: (N_pts, T, 9) = (N_pts, 9, 9) @ (9, T)  via einsum
    print("  Computing ||sigma||_F time series at all grid points...")
    t0 = time.perf_counter()
    # M_grid: (N_pts, 9, 9),  vec_A_body.T: (9, T)
    # result: (N_pts, T, 9)  -- do in chunks to avoid OOM
    CHUNK = 200
    frob_grid = np.zeros((N_pts, T))
    for start in range(0, N_pts, CHUNK):
        end       = min(start + CHUNK, N_pts)
        vs        = np.einsum('pqr,tr->ptq',
                              M_grid[start:end].reshape(end-start, 9, 9),
                              vec_A_body)         # (chunk, T, 9)
        frob_grid[start:end] = np.linalg.norm(vs, axis=2)
    print(f"  Done in {time.perf_counter()-t0:.2f}s")

    # For each threshold, compute MFPT at each grid point and plot
    for thresh in THRESHOLDS:
        print(f"\n  Threshold c = {thresh}")
        mfpt_grid = np.full(N_pts, np.nan)

        for idx in range(N_pts):
            wts = recurrence_times(frob_grid[idx], t, thresh)
            if len(wts) >= 5:
                mfpt_grid[idx] = wts.mean()

        # Reshape to (N_THETA, N_PHI) for surface plot
        MFPT = mfpt_grid.reshape(N_THETA, N_PHI)
        X    = (a[0] * np.sin(TH) * np.cos(PH))
        Y    = (a[1] * np.sin(TH) * np.sin(PH))
        Z    = (a[2] * np.cos(TH))

        # Mask NaN with mean for colour continuity
        MFPT_plot = np.where(np.isnan(MFPT), np.nanmean(MFPT), MFPT)

        fig = plt.figure(figsize=(10, 7))
        ax  = fig.add_subplot(111, projection='3d')
        norm = plt.Normalize(vmin=np.nanpercentile(mfpt_grid, 5),
                             vmax=np.nanpercentile(mfpt_grid, 95))
        surf = ax.plot_surface(X, Y, Z,
                               facecolors=cm.plasma_r(norm(MFPT_plot)),
                               rstride=1, cstride=1,
                               linewidth=0, antialiased=True, shade=False)
        sm = cm.ScalarMappable(cmap='plasma_r', norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, shrink=0.5, label="MFPT (time units)")
        ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("x3")
        ax.set_title(f"MFPT  ||sigma||_F > {thresh}\n"
                     f"(NaN = fewer than 5 events at that point)")
        plt.tight_layout()
        fname = os.path.join(DIR_SURFACE,
                             f"mfpt_surface_c{thresh:.3g}.png")
        plt.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Saved {fname}")

        # Also save a flat (theta, phi) heatmap — easier to read
        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.pcolormesh(np.degrees(PH), np.degrees(TH), MFPT,
                           cmap='plasma_r', shading='auto')
        plt.colorbar(im, ax=ax, label="MFPT (time units)")
        ax.set_xlabel("phi (degrees)")
        ax.set_ylabel("theta (degrees)")
        ax.set_title(f"MFPT colourmap  ||sigma||_F > {thresh}  "
                     f"(theta-phi parametrisation)")
        plt.tight_layout()
        fname = os.path.join(DIR_SURFACE,
                             f"mfpt_flatmap_c{thresh:.3g}.png")
        plt.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Saved {fname}")

else:
    print("\nPart A skipped (THRESHOLDS = []). "
          "Set THRESHOLDS to produce surface colourmaps.")

# ===========================================================================
# PART B — ANALYTICAL PDF MAPPING  A-GMM  ->  ||sigma||_F
# ===========================================================================
 
print("\n" + "="*55)
print("PART B: Analytical PDF mapping  A-GMM -> ||sigma||_F")
print("="*55)
print("""
Three curves per surface point:
  1. Empirical KDE       -- ground truth from the actual time series
  2. Lab-frame GMM       -- lab-frame A-GMM pushed through M analytically.
                            Will NOT match if the ellipsoid rotates significantly,
                            because the body-frame A distribution differs from
                            the lab-frame one.
  3. Body-frame GMM      -- GMM fitted directly to A_body(t) = R^T A R,
                            then pushed through M analytically.
                            This removes the rotation as a source of error.
                            Should match well; any remaining gap is due to the
                            norm nonlinearity or GMM approximation quality.
  4. Fitted GMM          -- GMM fitted directly to the empirical ||sigma||_F.
                            Always matches by construction; not a prediction.
""")
 
# ── Fit GMM to body-frame A (done once, shared across surface points) ──────
 
print("  Fitting GMM to body-frame A components...")
n_k = 3   # components — same as N_GMM_COMP for consistency
 
# vec_A_body: (T, 9) — already computed above from R^T A R
mu_Ab_k  = np.zeros((n_k, 9))
var_Ab_k = np.zeros((n_k, 9))
w_Ab_k   = np.zeros((n_k, 9))
 
A_BODY_COMP_NAMES = [f"A_body_{i}{j}" for i in range(3) for j in range(3)]
 
for comp_idx in range(9):
    data = vec_A_body[:, comp_idx]
    if data.std() < 1e-10:
        mu_Ab_k[:, comp_idx]  = data.mean()
        var_Ab_k[:, comp_idx] = 1e-10
        w_Ab_k[:, comp_idx]   = 1.0 / n_k
        continue
    gmm_ab = GaussianMixture(n_components=n_k, random_state=0)
    gmm_ab.fit(data.reshape(-1, 1))
    # sort by mean for consistency
    order = np.argsort(gmm_ab.means_.ravel())
    w_Ab_k[:, comp_idx]  = gmm_ab.weights_[order]
    mu_Ab_k[:, comp_idx] = gmm_ab.means_.ravel()[order]
    var_Ab_k[:, comp_idx]= gmm_ab.covariances_.ravel()[order]
 
w_Ab_joint = w_Ab_k.mean(axis=1)
w_Ab_joint /= w_Ab_joint.sum()
 
# ── Load lab-frame A-GMM parameters ───────────────────────────────────────
 
df_a         = pd.read_csv(A_GMM_CSV)
a_comp_names = sorted(df_a["component"].unique())
n_k_lab      = int(df_a.groupby("component")["weight"].count().iloc[0])
 
mu_A_k  = np.zeros((n_k_lab, 9))
var_A_k = np.zeros((n_k_lab, 9))
w_A_k   = np.zeros((n_k_lab, 9))
 
for comp_idx, comp in enumerate(a_comp_names):
    grp = df_a[df_a["component"] == comp].sort_values("weight", ascending=False)
    for k in range(n_k_lab):
        w_A_k[k, comp_idx]   = grp.iloc[k]["weight"]
        mu_A_k[k, comp_idx]  = grp.iloc[k]["mean"]
        var_A_k[k, comp_idx] = grp.iloc[k]["variance"]
 
w_lab_joint = w_A_k.mean(axis=1)
w_lab_joint /= w_lab_joint.sum()
 
 
def analytical_frob_pdf(mu_A_components, var_A_components, w_joint,
                         M, n_per_comp, xs):
    """
    Compute the analytical ||sigma||_F PDF by propagating a diagonal GMM
    through the linear map vec(sigma) = M @ vec(A), then taking the norm.
 
    Parameters
    ----------
    mu_A_components  : (n_k, 9)  per-component means of vec(A)
    var_A_components : (n_k, 9)  per-component variances of vec(A)
    w_joint          : (n_k,)    mixture weights
    M                : (9, 9)    stress transfer matrix
    n_per_comp       : int       MC draws per component
    xs               : (N,)      grid for PDF evaluation
 
    Returns
    -------
    pdf : (N,)  normalised PDF on xs
    """
    n_k   = len(w_joint)
    pdf   = np.zeros_like(xs)
 
    for k in range(n_k):
        mu_s  = M @ mu_A_components[k]                        # (9,)
        cov_s = M @ np.diag(var_A_components[k]) @ M.T        # (9,9)
 
        # Ensure PSD
        eigvals = np.linalg.eigvalsh(cov_s)
        if eigvals.min() < 0:
            cov_s += np.eye(9) * (-eigvals.min() + 1e-10)
 
        samples = np.random.multivariate_normal(mu_s, cov_s, size=n_per_comp)
        frob_k  = np.linalg.norm(samples, axis=1)
 
        if frob_k.std() < 1e-10:
            continue
        pdf += w_joint[k] * gaussian_kde(frob_k)(xs)
 
    integral = np.trapz(pdf, xs)
    return pdf / integral if integral > 0 else pdf
 
 
# ── Per named surface point ────────────────────────────────────────────────
 
pdf_gmm_rows = []
N_ANALYTIC   = 200_000
n_per_comp   = N_ANALYTIC // n_k
 
for pt_idx, x_pt in enumerate(named_points):
    coord_str = f"({x_pt[0]:.1f}, {x_pt[1]:.1f}, {x_pt[2]:.1f})"
    print(f"\n  Surface point {pt_idx+1}  {coord_str}")
 
    M = build_stress_transfer_matrix(Ellipsoid, a, mu, x_pt)
 
    # Empirical ||sigma||_F from the actual time series
    frob_ts  = np.linalg.norm(vec_A_body @ M.T, axis=1)   # (T,)
    x_max    = frob_ts.max() * 1.1
    xs       = np.linspace(0, x_max, 600)
    kde_emp  = gaussian_kde(frob_ts)
    emp_pdf  = kde_emp(xs)
 
    print(f"    mean={frob_ts.mean():.4f}  std={frob_ts.std():.4f}")
 
    # Curve 2: lab-frame GMM propagated through M
    print("    Computing lab-frame analytical PDF...", flush=True)
    lab_pdf = analytical_frob_pdf(
        mu_A_k, var_A_k, w_lab_joint, M, n_per_comp, xs
    )
 
    # Curve 3: body-frame GMM propagated through M  (the correct one)
    print("    Computing body-frame analytical PDF...", flush=True)
    body_pdf = analytical_frob_pdf(
        mu_Ab_k, var_Ab_k, w_Ab_joint, M, n_per_comp, xs
    )
 
    # Curve 4: GMM fitted directly to empirical ||sigma||_F
    gmm = GaussianMixture(n_components=N_GMM_COMP, random_state=0)
    gmm.fit(frob_ts.reshape(-1, 1))
    gmm_pdf = np.exp(gmm.score_samples(xs.reshape(-1, 1)))
 
    # L2 errors
    l2_lab  = np.sqrt(np.trapz((emp_pdf - lab_pdf )**2, xs))
    l2_body = np.sqrt(np.trapz((emp_pdf - body_pdf)**2, xs))
    print(f"    L2 error (lab-frame  analytical): {l2_lab:.5f}")
    print(f"    L2 error (body-frame analytical): {l2_body:.5f}")
    if l2_body < l2_lab:
        print("    => Body-frame GMM is a better predictor "
              "(rotation matters here)")
    else:
        print("    => Lab-frame and body-frame agree "
              "(rotation has little effect)")
 
    # ── Three-panel plot ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"||sigma||_F PDF  --  surface point {pt_idx+1}  {coord_str}",
        fontsize=13,
    )
 
    # Panel 1: lab-frame analytical vs empirical
    ax = axes[0]
    ax.fill_between(xs, emp_pdf, alpha=0.2, color="steelblue")
    ax.plot(xs, emp_pdf,  color="steelblue", lw=1.5, label="Empirical KDE")
    ax.plot(xs, lab_pdf,  color="tomato",    lw=2.0, ls="--",
            label="Lab-frame GMM -> M")
    ax.set_title("Lab-frame GMM propagation\n"
                 "(fails if ellipsoid rotates)")
    ax.set_xlabel("||sigma||_F"); ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.text(0.97, 0.95, f"L2={l2_lab:.4f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9)
 
    # Panel 2: body-frame analytical vs empirical
    ax = axes[1]
    ax.fill_between(xs, emp_pdf, alpha=0.2, color="steelblue")
    ax.plot(xs, emp_pdf,  color="steelblue", lw=1.5, label="Empirical KDE")
    ax.plot(xs, body_pdf, color="seagreen",  lw=2.0, ls="--",
            label="Body-frame GMM -> M")
    ax.set_title("Body-frame GMM propagation\n"
                 "(removes rotation error)")
    ax.set_xlabel("||sigma||_F")
    ax.legend(fontsize=8)
    ax.text(0.97, 0.95, f"L2={l2_body:.4f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9)
 
    # Panel 3: fitted GMM (always matches — descriptive not predictive)
    ax = axes[2]
    ax.fill_between(xs, emp_pdf, alpha=0.2, color="steelblue")
    ax.plot(xs, emp_pdf, color="steelblue", lw=1.5, label="Empirical KDE")
    ax.plot(xs, gmm_pdf, color="purple",    lw=2.0, ls="--",
            label=f"GMM fit ({N_GMM_COMP} components)")
    ax.set_title("Fitted GMM\n(descriptive — always matches)")
    ax.set_xlabel("||sigma||_F")
    ax.legend(fontsize=8)
 
    plt.tight_layout()
    fname = os.path.join(DIR_PDF, f"pdf_mapping_point{pt_idx+1}.png")
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"    Saved {fname}")
 
    for k in range(N_GMM_COMP):
        pdf_gmm_rows.append({
            "surface_point":   pt_idx + 1,
            "coordinates":     coord_str,
            "gmm_component_k": k,
            "weight":          gmm.weights_[k],
            "mean":            gmm.means_[k, 0],
            "variance":        gmm.covariances_[k, 0, 0],
        })
 
pd.DataFrame(pdf_gmm_rows, columns=[
    "surface_point", "coordinates", "gmm_component_k",
    "weight", "mean", "variance",
]).to_csv(PDF_GMM_CSV, index=False)
print(f"\nSaved GMM parameters for ||sigma||_F -> {PDF_GMM_CSV}")

# ===========================================================================
# PART B2 — COMPONENT-WISE VALIDATION  (no norm — should match exactly)
#
# For each surface point and each sigma_ij component:
#   sigma_ij(t) = M[p, :] @ vec(A_body(t))     purely linear
#
# The body-frame GMM prediction for sigma_ij is a 1-D GMM with:
#   weight_k   = w_Ab_joint[k]
#   mean_k     = M[p, :] @ mu_Ab_k[k]                 (scalar)
#   variance_k = M[p, :] @ diag(var_Ab_k[k]) @ M[p, :].T   (scalar)
#
# No Monte Carlo needed — this is analytically exact.
# Any mismatch with the empirical KDE reveals GMM approximation error only,
# with no contribution from the norm nonlinearity or the rotation.
# ===========================================================================
 
print("\n" + "="*55)
print("PART B2: Component-wise analytical validation")
print("  (linear map only — body-frame GMM should match exactly)")
print("="*55)
 
from scipy.stats import norm as sp_norm
 
SIGMA_LABELS = [
    [r"$\sigma_{11}$", r"$\sigma_{12}$", r"$\sigma_{13}$"],
    [r"$\sigma_{21}$", r"$\sigma_{22}$", r"$\sigma_{23}$"],
    [r"$\sigma_{31}$", r"$\sigma_{32}$", r"$\sigma_{33}$"],
]
 
compwise_rows = []
 
for pt_idx, x_pt in enumerate(named_points):
    coord_str = f"({x_pt[0]:.1f}, {x_pt[1]:.1f}, {x_pt[2]:.1f})"
    print(f"\n  Surface point {pt_idx+1}  {coord_str}")
 
    M = build_stress_transfer_matrix(Ellipsoid, a, mu, x_pt)   # (9, 9)
    vec_sigma_ts = vec_A_body @ M.T                             # (T, 9)
 
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(
        f"Component-wise sigma_ij PDF — surface point {pt_idx+1}  {coord_str}\n"
        f"Body-frame GMM prediction vs empirical KDE  "
        f"(linear map: should match perfectly)",
        fontsize=12,
    )
 
    for p in range(9):
        i, j  = p // 3, p % 3
        ax    = axes[i, j]
        data  = vec_sigma_ts[:, p]
        std_d = data.std()
 
        if std_d < 1e-10:
            ax.text(0.5, 0.5, f"constant\n≈ {data.mean():.3g}",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="grey")
            ax.set_title(SIGMA_LABELS[i][j], fontsize=11)
            continue
 
        spread = data.max() - data.min()
        xs_c   = np.linspace(data.min() - 0.15*spread,
                             data.max() + 0.15*spread, 500)
 
        # Empirical KDE
        kde_c = gaussian_kde(data)
        emp_c = kde_c(xs_c)
        ax.fill_between(xs_c, emp_c, alpha=0.2, color="steelblue")
        ax.plot(xs_c, emp_c, color="steelblue", lw=1.5, label="Empirical KDE")
 
        # Analytical body-frame GMM — exact, no MC
        m_row    = M[p, :]
        pred_pdf = np.zeros_like(xs_c)
        for k in range(n_k):
            mu_k  = float(m_row @ mu_Ab_k[k])
            var_k = float(m_row @ (var_Ab_k[k] * m_row))
            var_k = max(var_k, 1e-12)
            pred_pdf += w_Ab_joint[k] * sp_norm.pdf(xs_c, mu_k, np.sqrt(var_k))
 
        ax.plot(xs_c, pred_pdf, color="seagreen", lw=2.0, ls="--",
                label="Body-frame GMM (analytical)")
 
        l2 = np.sqrt(np.trapz((emp_c - pred_pdf)**2, xs_c))
        ax.text(0.97, 0.95, f"L2={l2:.4f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8)
        ax.set_title(SIGMA_LABELS[i][j], fontsize=11)
        ax.set_xlabel("stress value", fontsize=8)
        ax.set_ylabel("density",      fontsize=8)
        if p == 0:
            ax.legend(fontsize=7)
 
        compwise_rows.append({
            "surface_point":  pt_idx + 1,
            "coordinates":    coord_str,
            "component":      f"sigma_{i}{j}",
            "l2_error":       l2,
            "empirical_mean": float(data.mean()),
            "empirical_std":  float(std_d),
            "predicted_mean": float(sum(w_Ab_joint[k] * float(M[p,:] @ mu_Ab_k[k])
                                        for k in range(n_k))),
        })
 
    plt.tight_layout()
    fname = os.path.join(DIR_PDF, f"pdf_componentwise_point{pt_idx+1}.png")
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"    Saved {fname}")
 
    df_cw = pd.DataFrame(compwise_rows)
    df_pt = df_cw[df_cw["surface_point"] == pt_idx + 1]
    print(f"    L2 errors per component:")
    for ii in range(3):
        row_str = "    "
        for jj in range(3):
            comp = f"sigma_{ii}{jj}"
            val  = df_pt[df_pt["component"] == comp]["l2_error"].values
            row_str += f"  {comp}: {val[0]:.4f}" if len(val) else f"  {comp}: ---"
        print(row_str)
 
pd.DataFrame(compwise_rows).to_csv(
    os.path.join(DIR_PDF, "componentwise_l2.csv"), index=False
)
print(f"\nSaved component-wise L2 errors -> {DIR_PDF}/componentwise_l2.csv")
 
print("\n" + "="*55)
print("Done. Outputs:")
if THRESHOLDS:
    print(f"  {DIR_SURFACE}/  -- MFPT surface colourmaps and flatmaps")
print(f"  {DIR_PDF}/       -- analytical vs empirical PDF figures")
print(f"  {PDF_GMM_CSV}    -- GMM fit to ||sigma||_F per surface point")
print("="*55)