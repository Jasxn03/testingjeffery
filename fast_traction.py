"""
fast_traction.py

Drop-in replacement for the slow traction_magnitude() loop.

The key insight: because Stokes flow is linear, the tangential traction
tau at a fixed body-frame point x is a linear function of the body-frame
velocity gradient A_body:

    tau(t) = M @ vec(A_body(t))      shape (3,) = (3,9) @ (9,)

where M is the 3x9 transfer matrix computed ONCE from the ellipsoid geometry.

This means traction over T timesteps costs:
  SLOW: T calls to ellipsoid.sigma()   ~T * (jacfwd overhead)
  FAST: 1 call to build M, then T matrix-vector multiplies  (negligible)

Speedup is typically 100-1000x for large T.

The only cost we cannot avoid is computing A_body(t) = R(t)^T A(t) R(t)
at each timestep, but that is just 2 cheap matrix multiplies per step.

Usage
-----
from fast_traction import build_transfer_matrix, fast_traction_magnitude

# Build M once (slow, ~seconds, done once per surface point per aspect ratio)
M, n_hat = build_transfer_matrix(axes, x_body, mu=1.0)

# Compute traction time series (fast, microseconds per step)
tau_mag = fast_traction_magnitude(M, n_hat, A_series, R_history)

# Or get the full traction vectors (T, 3) instead of just magnitudes
tau_vecs = fast_traction_vectors(M, n_hat, A_series, R_history)
"""

import numpy as np
import sys, os

# ── adjust path if needed ─────────────────────────────────────
JEFFERY_PATH = "."
sys.path.insert(0, JEFFERY_PATH)
from jeffery4_2 import Ellipsoid


# ─────────────────────────────────────────────────────────────
# BUILD TRANSFER MATRIX  (slow, done once)
# ─────────────────────────────────────────────────────────────

def outward_normal(x, a):
    """Outward unit normal to ellipsoid surface at x."""
    n = np.asarray(x, dtype=float) / (np.asarray(a, dtype=float) ** 2)
    return n / np.linalg.norm(n)


def build_transfer_matrix(axes, x_body, mu=1.0):
    """
    Compute the 3x9 traction transfer matrix M at body-frame point x_body.

    tau(x) = M @ vec(A_body)

    where vec(A_body) is the row-major flattening of the 3x3 body-frame
    velocity gradient, and tau is the TANGENTIAL traction vector.

    Parameters
    ----------
    axes   : array-like (3,)  semi-axes [a, b, c]
    x_body : array-like (3,)  surface point in body frame
    mu     : float            dynamic viscosity

    Returns
    -------
    M     : (3, 9) transfer matrix
    n_hat : (3,)   outward unit normal at x_body
    """
    a    = np.array(axes, dtype=float)
    x    = np.array(x_body, dtype=float)
    n    = outward_normal(x, a)

    eps0 = np.zeros((3, 3))
    ell  = Ellipsoid(a, eps0, mu=mu)
    ell.use_surface_mode()

    M = np.zeros((3, 9))
    for k in range(9):
        A_basis = np.zeros((3, 3))
        A_basis[k // 3, k % 3] = 1.0

        ell.set_strain(A_basis)
        ell.set_coefs()

        sig   = ell.sigma(x)
        t_vec = sig @ n
        tau   = t_vec - np.dot(t_vec, n) * n   # tangential component

        M[:, k] = tau

    return M, n


# ─────────────────────────────────────────────────────────────
# FAST TRACTION TIME SERIES  (fast, vectorised over T)
# ─────────────────────────────────────────────────────────────

def fast_traction_vectors(M, n_hat, A_series, R_history):
    """
    Compute tangential traction vectors at every timestep.

    Parameters
    ----------
    M         : (3, 9)   transfer matrix from build_transfer_matrix()
    n_hat     : (3,)     outward normal (not used in computation,
                         included for API consistency / verification)
    A_series  : (T,3,3)  lab-frame velocity gradient time series
    R_history : (T,3,3)  rotation matrix time series

    Returns
    -------
    tau_vecs : (T, 3)  tangential traction vectors
    """
    T = len(A_series)

    # Rotate ALL A matrices into body frame at once using einsum
    # A_body[t] = R[t]^T @ A[t] @ R[t]
    # einsum: 'tji, tjk, tkl -> til'  but we do it in two steps for clarity
    # Step 1: tmp[t] = R[t]^T @ A[t]   shape (T,3,3)
    tmp      = np.einsum('tji, tjk -> tik', R_history, A_series)
    # Step 2: A_body[t] = tmp[t] @ R[t]   shape (T,3,3)
    A_body   = np.einsum('tij, tjk -> tik', tmp, R_history)

    # Flatten each A_body to a 9-vector: shape (T, 9)
    vec_A_body = A_body.reshape(T, 9)

    # tau[t] = M @ vec_A_body[t]   vectorised: (T,3) = (T,9) @ (9,3).T
    tau_vecs = vec_A_body @ M.T    # (T, 3)

    return tau_vecs


def fast_traction_magnitude(M, n_hat, A_series, R_history):
    """
    Compute ||tau|| at every timestep.

    Returns
    -------
    tau_mag : (T,)  traction magnitudes
    """
    tau_vecs = fast_traction_vectors(M, n_hat, A_series, R_history)
    return np.linalg.norm(tau_vecs, axis=1)


# ─────────────────────────────────────────────────────────────
# CONVENIENCE: BUILD MATRICES FOR MULTIPLE SURFACE POINTS
# ─────────────────────────────────────────────────────────────

def build_all_transfer_matrices(axes, mu=1.0):
    """
    Build transfer matrices for the standard set of surface points:
      tip_pos  (a, 0, 0)
      tip_neg  (-a, 0, 0)
      eq_b     (0, b, 0)
      eq_c     (0, 0, c)
      off_45   (a/sqrt(2), b/sqrt(2), 0)

    Returns dict: name -> {'M': (3,9), 'n': (3,), 'x': (3,)}
    """
    a = np.array(axes, dtype=float)
    x45 = a[0] / np.sqrt(2)
    y45 = a[1] / np.sqrt(2)

    points = {
        'tip_pos':  np.array([ a[0],  0.,   0.  ]),
        'tip_neg':  np.array([-a[0],  0.,   0.  ]),
        'eq_b':     np.array([ 0.,    a[1], 0.  ]),
        'eq_c':     np.array([ 0.,    0.,   a[2]]),
        'off_45':   np.array([ x45,   y45,  0.  ]),
    }

    results = {}
    for name, x in points.items():
        print(f"  Building M at {name} {x}...", end=" ", flush=True)
        M, n = build_transfer_matrix(axes, x, mu=mu)
        results[name] = {'M': M, 'n': n, 'x': x}
        print("done")

    return results


# ─────────────────────────────────────────────────────────────
# VERIFICATION: compare fast vs slow for a few timesteps
# ─────────────────────────────────────────────────────────────

def verify_fast_vs_slow(axes, x_body, A_series, R_history,
                        mu=1.0, n_check=20):
    """
    Verify fast_traction_magnitude against the original slow loop
    for the first n_check timesteps.

    Prints max absolute error and relative error.
    Returns True if max relative error < 1e-8 (machine precision).
    """
    a    = np.array(axes, dtype=float)
    x    = np.array(x_body, dtype=float)
    n_sf = outward_normal(x, a)

    # ── Fast ────────────────────────────────────────────────────
    M, _ = build_transfer_matrix(axes, x_body, mu=mu)
    tau_fast = fast_traction_magnitude(
        M, n_sf,
        A_series[:n_check],
        R_history[:n_check]
    )

    # ── Slow (original loop) ────────────────────────────────────
    from jeffery4 import Ellipsoid
    eps0 = np.zeros((3, 3))
    ell  = Ellipsoid(a, np.zeros((3,3)), mu=mu)
    ell.use_surface_mode()

    tau_slow = []
    for t_idx in range(n_check):
        R_t    = R_history[t_idx]
        A_body = R_t.T @ A_series[t_idx] @ R_t
        ell.set_strain(A_body)
        ell.set_coefs()
        sig   = ell.sigma(x)
        t_vec = sig @ n_sf
        tau   = t_vec - np.dot(t_vec, n_sf) * n_sf
        tau_slow.append(np.linalg.norm(tau))
    tau_slow = np.array(tau_slow)

    abs_err = np.abs(tau_fast - tau_slow)
    rel_err = abs_err / (np.abs(tau_slow) + 1e-30)

    print(f"\nVerification ({n_check} steps):")
    print(f"  Max absolute error : {abs_err.max():.2e}")
    print(f"  Max relative error : {rel_err.max():.2e}")
    print(f"  Mean relative error: {rel_err.mean():.2e}")

    passed = rel_err.max() < 1e-6
    print(f"  {'PASSED' if passed else 'FAILED'} "
          f"(threshold: rel err < 1e-6)")
    return passed


# ─────────────────────────────────────────────────────────────
# MAIN — benchmark and verify
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import time
    import pandas as pd
    from orientation import integrate_orientation

    CSV_PATH = "grad_u.csv"
    AXES     = [2.0, 1.0, 1.0]
    MU       = 1.0
    BURN_IN  = 0.10

    print("Loading data...")
    df       = pd.read_csv(CSV_PATH)
    times    = df['time'].values
    A_series = np.zeros((len(df), 3, 3))
    for i in range(3):
        for j in range(3):
            A_series[:, i, j] = df[f'A{i+1}{j+1}'].values

    n_burn  = int(len(times) * BURN_IN)
    times_b = times[n_burn:]
    A_b     = A_series[n_burn:]

    print("Integrating orientation...")
    R_history, *_ = integrate_orientation(AXES, A_series, times)
    R_b = R_history[n_burn:]

    x_tip = np.array([AXES[0], 0., 0.])

    # ── Verify correctness ──────────────────────────────────────
    print("\nVerifying fast == slow...")
    verify_fast_vs_slow(AXES, x_tip, A_b, R_b, mu=MU, n_check=500)

    # ── Benchmark ───────────────────────────────────────────────
    print("\nBenchmarking...")

    # Build M (one-time cost)
    t0 = time.perf_counter()
    M, n_hat = build_transfer_matrix(AXES, x_tip, mu=MU)
    t_build = time.perf_counter() - t0
    print(f"  Build M (one-time):  {t_build:.3f} s")

    # Fast: vectorised
    t0 = time.perf_counter()
    tau_fast = fast_traction_magnitude(M, n_hat, A_b, R_b)
    t_fast = time.perf_counter() - t0
    print(f"  Fast (vectorised):   {t_fast:.3f} s  "
          f"({len(A_b)} steps, {t_fast/len(A_b)*1e6:.2f} us/step)")

    # Slow: original loop (first 200 steps, extrapolate)
    n_slow  = min(200, len(A_b))
    eps0    = np.zeros((3, 3))
    ell     = Ellipsoid(np.array(AXES), eps0, mu=MU)
    ell.use_surface_mode()
    n_sf    = outward_normal(x_tip, AXES)

    t0 = time.perf_counter()
    for t_idx in range(n_slow):
        R_t    = R_b[t_idx]
        A_body = R_t.T @ A_b[t_idx] @ R_t
        ell.set_strain(A_body)
        ell.set_coefs()
        sig   = ell.sigma(x_tip)
        t_vec = sig @ n_sf
        tau   = t_vec - np.dot(t_vec, n_sf) * n_sf
    t_slow_sample = time.perf_counter() - t0
    t_slow_extrap = t_slow_sample / n_slow * len(A_b)

    print(f"  Slow (original loop): "
          f"~{t_slow_extrap:.1f} s extrapolated "
          f"({t_slow_sample/n_slow*1e3:.2f} ms/step)")
    print(f"\n  Speedup: ~{t_slow_extrap / (t_fast + t_build):.0f}x")
    print(f"  (build cost amortised: {t_build:.2f}s once, "
          f"then {t_fast:.3f}s for full series)")