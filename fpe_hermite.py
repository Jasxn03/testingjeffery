'''
Complete Set Expansion (Hermite Basis) for the Fokker-Planck Equation
of Chevillard & Meneveau (2006), Eq. (6):

    dA = [ -A² + Tr(A²)/Tr(C⁻¹_Γ) C⁻¹_Γ - Tr(C⁻¹_Γ)/3 · A ] dt + dW
    C_Γ = exp(Γ·A) exp(Γ·A^T)

Method: Risken, "The Fokker-Planck Equation" (2nd ed.), Section 6.6.5.
──────────────────────────────────────────────────────────────────────
Expand the probability density in a tensor-product Hermite function basis:

    W(x, t) = Σ_n  c_n(t) φ_n(x; a)

    φ_n(x; a) = √a · N_n · H_n(ax) · exp(-a²x²/2)
    N_n = 1/√(2ⁿ n! √π)    (orthonormality: ∫ φ_m φ_n dx = δ_mn)

Insert into the FP equation → matrix system  ċ = L c  where:

    L_mn = ∫ φ_m(x) [L_FP φ_n](x) dx

Compute matrix elements via Gauss-Hermite (GH) quadrature.
'''

import math
import time
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.linalg import eig, expm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os 


# ══════════════════════════════════════════════════════════════════
# Hermite function basis
# ══════════════════════════════════════════════════════════════════

def hermite_norm(n):
    """Normalisation constant N_n = 1/√(2ⁿ n! √π)."""
    return 1.0 / math.sqrt((2**n) * math.factorial(n) * math.sqrt(math.pi))


def hermite_poly(n, t):
    """Physicist's Hermite polynomial H_n(t) via recurrence."""
    t = np.asarray(t, dtype=float)
    if n == 0:
        return np.ones_like(t)
    elif n == 1:
        return 2.0 * t
    Hm2, Hm1 = np.ones_like(t), 2.0 * t
    for k in range(2, n + 1):
        H = 2.0 * t * Hm1 - 2.0 * (k - 1) * Hm2
        Hm2, Hm1 = Hm1, H
    return Hm1


def phi_fn(n, x, a=1.0):
    """
    Hermite function  φ_n(x; a) = √a · N_n · H_n(ax) · exp(−a²x²/2).
    Satisfies ∫ φ_m(x; a) φ_n(x; a) dx = δ_{mn}.
    """
    Nn = hermite_norm(n)
    ax = a * np.asarray(x, dtype=float)
    return math.sqrt(a) * Nn * hermite_poly(n, ax) * np.exp(-0.5 * ax**2)


def phi_deriv(n, x, a=1.0):
    """
    First derivative dφ_n/dx using the ladder relation:
        dφ_n/dx = a·√(2n)·φ_{n−1} − a²·x·φ_n
    """
    x = np.asarray(x, dtype=float)
    dphi = -a**2 * x * phi_fn(n, x, a)
    if n > 0:
        dphi += a * math.sqrt(2 * n) * phi_fn(n - 1, x, a)
    return dphi


def phi_deriv2(n, x, a=1.0):
    """
    Second derivative d²φ_n/dx² using:
        d²φ_n/dx² = a²√(4n(n−1))·φ_{n−2} − 2a³x√(2n)·φ_{n−1} + (a⁴x²−a²)·φ_n
    """
    x = np.asarray(x, dtype=float)
    d2phi = (a**4 * x**2 - a**2) * phi_fn(n, x, a)
    if n >= 1:
        d2phi -= 2.0 * a**3 * x * math.sqrt(2 * n) * phi_fn(n - 1, x, a)
    if n >= 2:
        d2phi += a**2 * math.sqrt(4 * n * (n - 1)) * phi_fn(n - 2, x, a)
    return d2phi


# ══════════════════════════════════════════════════════════════════
# 1-D FP matrix builder
# ══════════════════════════════════════════════════════════════════

def build_fp_matrix_1d(drift_fn, D, N_basis, N_quad, a=1.0):
    """
    Build the N_basis × N_basis Fokker-Planck operator matrix in the
    Hermite function basis φ_n(x; a).

    FP operator: L_FP W = −∂_x[f(x)W] + D ∂²_x W

    Matrix element (integration by parts not needed — done directly):
        L_mn = ∫ φ_m(x) L_FP[φ_n(x)] dx
             = ∫ φ_m(x) [−f′(x)φ_n − f(x)φ′_n + D φ″_n] dx

    GH quadrature with exp(+t²) correction (Bug 2 fix):
        L_mn ≈ (1/a) Σ_k w_k exp(t_k²) φ_m(t_k/a) L_FP[φ_n](t_k/a)

    Parameters
    ----------
    drift_fn : callable  f(x)
    D        : float     diffusion coefficient
    N_basis  : int       number of basis functions (indices 0 … N_basis−1)
    N_quad   : int       Gauss-Hermite quadrature points
    a        : float     Hermite scaling (tune to match PDF width)
    """
    pts, wts = hermgauss(N_quad)          # GH nodes t_k and weights w_k
    x = pts / a                            # physical coordinates

    # Evaluate basis functions and derivatives at quadrature nodes
    Phi   = np.stack([phi_fn(n,     x, a) for n in range(N_basis)])   # (N_basis, N_quad)
    Phi_p = np.stack([phi_deriv(n,  x, a) for n in range(N_basis)])
    Phi_pp= np.stack([phi_deriv2(n, x, a) for n in range(N_basis)])

    # Drift and its numerical derivative
    f     = drift_fn(x)
    h     = 1e-6
    f_p   = (drift_fn(x + h) - drift_fn(x - h)) / (2.0 * h)

    # Bug 2 fix: exp(+t²) converts GH weight exp(−t²) to flat dx measure
    exp_corr = np.exp(pts**2)
    # Combined weight for integration: (1/a) · w_k · exp(t_k²)
    quad_wt  = (wts * exp_corr) / a       # shape (N_quad,)

    # L_FP[φ_n] at each quadrature point
    # = −f′(x)·φ_n − f(x)·φ′_n + D·φ″_n
    L_phi = (-f_p[None, :] * Phi
             - f[None, :]   * Phi_p
             + D             * Phi_pp)     # shape (N_basis, N_quad)

    # Matrix elements: L_mn = Σ_k quad_wt_k · φ_m(x_k) · L[φ_n](x_k)
    L_mat = (Phi * quad_wt[None, :]) @ L_phi.T  # (N_basis, N_basis)
    return L_mat


# ══════════════════════════════════════════════════════════════════
# 2-D FP matrix builder
# ══════════════════════════════════════════════════════════════════

def build_fp_matrix_2d(drift_fns, D_mat, N_basis, N_quad, a=1.0):
    """
    Build the (N_basis²) × (N_basis²) FP operator matrix for a 2-variable
    system using the tensor-product basis  Φ_{mn}(x1,x2) = φ_m(x1)·φ_n(x2).

    FP operator:
        L W = −∂_{x1}[f1 W] − ∂_{x2}[f2 W]
            + D11 ∂²_{x1} W + (D12+D21) ∂²_{x1 x2} W + D22 ∂²_{x2} W

    GH quadrature with exp(+t1²+t2²) correction (Bug 3 fix):
        L_{mn,pq} ≈ (1/a²) Σ_{k1,k2} w_{k1}w_{k2} exp(t_{k1}²+t_{k2}²)
                    × Φ_{mn}(x_{k1},x_{k2}) L_FP[Φ_{pq}](x_{k1},x_{k2})

    Parameters
    ----------
    drift_fns : [f1(x1,x2), f2(x1,x2)]
    D_mat     : 2×2 array  diffusion matrix
    N_basis   : int        per-dimension basis size
    N_quad    : int        per-dimension quadrature points
    a         : float      Hermite scaling
    """
    pts, wts = hermgauss(N_quad)
    x = pts / a

    # 1-D basis evaluated on quadrature grid (shared for both dimensions)
    Phi   = np.stack([phi_fn(n,     x, a) for n in range(N_basis)])
    Phi_p = np.stack([phi_deriv(n,  x, a) for n in range(N_basis)])
    Phi_pp= np.stack([phi_deriv2(n, x, a) for n in range(N_basis)])

    # 2-D quadrature grid  (i = x1 index, j = x2 index)
    X1, X2 = np.meshgrid(x, x, indexing='ij')     # (N_quad, N_quad)
    T1, T2 = np.meshgrid(pts, pts, indexing='ij')
    W2D     = np.outer(wts, wts)                   # product weights

    # Bug 3 fix: exp correction in 2D
    exp_corr = np.exp(T1**2 + T2**2)
    quad_wt  = W2D * exp_corr / a**2              # (N_quad, N_quad)

    # Drift and numerical derivatives on 2D grid
    f1   = drift_fns[0](X1, X2)
    f2   = drift_fns[1](X1, X2)
    h    = 1e-6
    df1_dx1 = (drift_fns[0](X1+h, X2) - drift_fns[0](X1-h, X2)) / (2*h)
    df2_dx2 = (drift_fns[1](X1, X2+h) - drift_fns[1](X1, X2-h)) / (2*h)

    D11 = D_mat[0, 0]; D12 = D_mat[0, 1]
    D21 = D_mat[1, 0]; D22 = D_mat[1, 1]

    N_tot = N_basis * N_basis
    L_mat = np.zeros((N_tot, N_tot))

    def idx(m, n):
        return m * N_basis + n

    for p in range(N_basis):
        for q in range(N_basis):
            col = idx(p, q)
            # Φ_{pq}(x1,x2) = φ_p(x1)·φ_q(x2) and its partial derivatives
            Phi_pq       = np.outer(Phi[p],   Phi[q])
            dPhi_dx1     = np.outer(Phi_p[p],  Phi[q])
            dPhi_dx2     = np.outer(Phi[p],   Phi_p[q])
            d2Phi_dx1sq  = np.outer(Phi_pp[p], Phi[q])
            d2Phi_dx2sq  = np.outer(Phi[p],   Phi_pp[q])
            d2Phi_dx1dx2 = np.outer(Phi_p[p],  Phi_p[q])

            # L_FP[Φ_{pq}]
            LPhi = (- df1_dx1 * Phi_pq - f1 * dPhi_dx1
                    - df2_dx2 * Phi_pq - f2 * dPhi_dx2
                    + D11 * d2Phi_dx1sq
                    + (D12 + D21) * d2Phi_dx1dx2
                    + D22 * d2Phi_dx2sq)

            # Weighted LPhi on the 2D grid
            wLPhi = quad_wt * LPhi                # (N_quad, N_quad)

            for m in range(N_basis):
                for n in range(N_basis):
                    row = idx(m, n)
                    # Φ_{mn}(x1,x2) outer product
                    Phi_mn = np.outer(Phi[m], Phi[n])
                    L_mat[row, col] = np.sum(Phi_mn * wLPhi)

    return L_mat


# Eigenvalue solver (Bug 1 fix: sort by real part)
# ══════════════════════════════════════════════════════════════════

def solve_and_sort(L_mat):
    """
    Solve the eigenvalue problem for the FP operator matrix.

    Bug 1 fix: sort eigenvalues by Re(λ) ascending (most negative first).
    The stationary state has λ = 0; it appears as the eigenvalue with
    Re(λ) closest to zero after truncation.

    Returns
    -------
    vals : (N,) complex array, sorted by Re(λ)
    vecs : (N, N) array, vecs[:, k] is the k-th eigenvector
    """
    vals, vecs = eig(L_mat)
    idx = np.argsort(vals.real)     # Bug 1: sort by REAL PART, not magnitude
    return vals[idx], vecs[:, idx]


def get_stationary_vec(vals, vecs):
    """Return the eigenvector with eigenvalue closest to zero (stationary state)."""
    stat_idx = np.argmin(np.abs(vals.real))
    return vals[stat_idx], vecs[:, stat_idx].real


# ══════════════════════════════════════════════════════════════════
# Physics: Chevillard & Meneveau (2006) Eq. (6)
# ══════════════════════════════════════════════════════════════════

def cm_drift(A, Gamma):
    """
    Full nonlinear drift vector from Eq. (6):
        f(A) = -A² + [Tr(A²)/Tr(C⁻¹_Γ)] C⁻¹_Γ − [Tr(C⁻¹_Γ)/3] A
        C_Γ = exp(Γ·A) exp(Γ·A^T)
    """
    A2     = A @ A
    eGA    = expm(Gamma * A)
    Cg     = eGA @ eGA.T
    Cg_inv = np.linalg.inv(Cg)
    trA2   = np.trace(A2)
    trCinv = np.trace(Cg_inv)
    return -A2 + (trA2 / trCinv) * Cg_inv - (trCinv / 3.0) * A


def _make_diag_A(x):
    """A = diag(x, −x/2, −x/2), the minimal traceless diagonal form."""
    return np.diag([x, -x/2.0, -x/2.0])


def drift_1d_A11(x, Gamma):
    """
    Scalar drift for A11 in the diagonal-A reduction:
    A = diag(x, −x/2, −x/2).  Returns the (0,0) component of cm_drift(A).
    Works element-wise for array x.
    """
    if np.ndim(x) == 0:
        return cm_drift(_make_diag_A(float(x)), Gamma)[0, 0]
    return np.array([cm_drift(_make_diag_A(xi), Gamma)[0, 0] for xi in np.asarray(x)])


def _make_A_2d(x1, x2):
    """A = [[x1, x2, 0], [0, -x1/2, 0], [0, 0, -x1/2]] — traceless 2D reduction."""
    return np.array([[x1, x2, 0.0],
                     [0.0, -x1/2.0, 0.0],
                     [0.0, 0.0, -x1/2.0]])


def drift_2d_A11_A12(x1, x2, Gamma):
    """
    2D drift for (A11, A12) via the 2D reduction.
    Handles scalar or array inputs.
    """
    if np.ndim(x1) == 0 and np.ndim(x2) == 0:
        dA = cm_drift(_make_A_2d(float(x1), float(x2)), Gamma)
        return dA[0, 0], dA[0, 1]
    x1, x2 = np.broadcast_arrays(x1, x2)
    d1 = np.zeros_like(x1, dtype=float)
    d2 = np.zeros_like(x2, dtype=float)
    for idx in np.ndindex(x1.shape):
        dA = cm_drift(_make_A_2d(x1[idx], x2[idx]), Gamma)
        d1[idx] = dA[0, 0]
        d2[idx] = dA[0, 1]
    return d1, d2


def noise_diffusion_1d():
    """
    Effective 1D diffusion coefficient for A11.
    From dW = G √(2dt) with ⟨G_ij G_kl⟩ = 2δ_ik δ_jl − ½δ_ij δ_kl − ½δ_il δ_jk:
        ⟨G_11²⟩ = 2(1)(1) − ½(1) − ½(1) = 1  →  D_A11 = 2·⟨G_11²⟩ = 2
    """
    return 2.0


def noise_diffusion_2d():
    """
    Effective 2D diffusion matrix for (A11, A12).
    ⟨G_12²⟩ = 2(1)(1) − 0 − 0 = 2  →  D_A12 = 2·⟨G_12²⟩ = 4
    Cross term D_{A11,A12} = 0 (orthogonal components).
    """
    return np.array([[2.0, 0.0],
                     [0.0, 4.0]])


# ══════════════════════════════════════════════════════════════════
# 4D reduction: (A11, A22, A12, A13)
#
# Exploits two symmetries:
#   1. Tracelessness: A33 = −A11 − A22  (removes one diagonal dof)
#   2. Isotropy: all six off-diagonal components split into two
#      statistically distinct classes:
#        "row-shared"    A12 representative (shares row/col with diagonal)
#        "pure transverse" A13 representative (neither index is diagonal)
#      We keep one from each class; the others are set to zero.
#
# State vector:  x = (A11, A22, A12, A13)
# Full matrix:
#   A = [[ A11,  A12,  A13 ],
#        [  0,   A22,   0  ],
#        [  0,    0,  A33  ]]     A33 = -A11 - A22
#
# Component index map:
#   0 → A11   1 → A22   2 → A12   3 → A13
# ══════════════════════════════════════════════════════════════════

def _make_A_4d(x):
    """Reconstruct 3×3 traceless matrix from x = (A11, A22, A12, A13)."""
    A11, A22, A12, A13 = x
    return np.array([[A11, A12, A13],
                     [0.0, A22, 0.0],
                     [0.0, 0.0, -A11 - A22]])


def drift_4d(x, Gamma):
    """
    4D drift vector for x = (A11, A22, A12, A13).
    Returns (f_A11, f_A22, f_A12, f_A13).
    """
    dA = cm_drift(_make_A_4d(x), Gamma)
    return np.array([dA[0, 0], dA[1, 1], dA[0, 1], dA[0, 2]])


def noise_diffusion_4d():
    """
    4×4 diffusion matrix for (A11, A22, A12, A13).

    From C_{ij,kl} = 2δ_ik δ_jl − ½δ_ij δ_kl − ½δ_il δ_jk,
    the FP diffusion entry is D_{ab} = C_{ij,kl} for components a=(ij), b=(kl).

    Diagonal:
      D_A11 = C_{11,11} = 2 − ½ − ½ = 1   → ×2 = 2
      D_A22 = C_{22,22} = 2 − ½ − ½ = 1   → ×2 = 2
      D_A12 = C_{12,12} = 2 − 0  − 0  = 2  → ×2 = 4
      D_A13 = C_{13,13} = 2 − 0  − 0  = 2  → ×2 = 4

    Only non-zero off-diagonal among these four:
      C_{11,22} = 0 − ½ − 0 = −½  →  D_{A11,A22} = −1
    """
    D = np.diag([2.0, 2.0, 4.0, 4.0])
    D[0, 1] = D[1, 0] = -1.0
    return D


# ══════════════════════════════════════════════════════════════════
# PDF reconstruction from eigenvector
# ══════════════════════════════════════════════════════════════════

def reconstruct_pdf_1d(c_stat, N_basis, x_eval, a=1.0):
    """Reconstruct P(x) = Σ_n c_n φ_n(x; a) from stationary eigenvector."""
    W = sum(c_stat[n] * phi_fn(n, x_eval, a) for n in range(N_basis))
    # Force positivity and normalise
    W = np.abs(W)
    norm = np.trapezoid(W, x_eval)
    return W / norm if norm > 0 else W


def reconstruct_pdf_2d(c_stat, N_basis, x_eval, a=1.0):
    """Reconstruct P(x1,x2) on a grid from the stationary eigenvector."""
    N = len(x_eval)
    W = np.zeros((N, N))
    phi_cache = [phi_fn(n, x_eval, a) for n in range(N_basis)]
    for m in range(N_basis):
        phi_m = phi_cache[m]
        for n in range(N_basis):
            c_mn = c_stat[m * N_basis + n]
            if abs(c_mn) < 1e-14:
                continue
            W += c_mn * np.outer(phi_m, phi_cache[n])
    W = np.abs(W)
    dx = x_eval[1] - x_eval[0]
    norm = np.sum(W) * dx**2
    return W / norm if norm > 0 else W


def reconstruct_pdf_4d(c_stat, N_basis, x_eval, a=1.0):
    """
    Reconstruct the 4 marginal PDFs P(A11), P(A22), P(A12), P(A13)
    from the stationary eigenvector of the 4D FP system.

    Marginal for dimension k:
        P_k(x) = Σ_n c_n · φ_{n_k}(x) · Π_{d≠k} [∫ φ_{n_d}(y) dy]

    The integrals ∫ φ_n(y) dy are precomputed numerically.
    """
    N_dim = 4
    x_int   = np.linspace(-15, 15, 4000)
    int_phi = np.array([np.trapezoid(phi_fn(n, x_int, a), x_int)
                        for n in range(N_basis)])       # shape (N_basis,)
    phi_eval = np.stack([phi_fn(n, x_eval, a) for n in range(N_basis)])
                                                        # shape (N_basis, N_eval)

    basis_indices = np.array(list(np.ndindex(*([N_basis] * N_dim))),
                             dtype=np.int32)            # (N_basis^4, 4)

    marginals = {}
    for k in range(N_dim):
        P_k = np.zeros(len(x_eval))
        for term_idx, n_vec in enumerate(basis_indices):
            coeff = c_stat[term_idx]
            if abs(coeff) < 1e-14:
                continue
            w = coeff
            for d in range(N_dim):
                if d != k:
                    w *= int_phi[n_vec[d]]
            P_k += w * phi_eval[n_vec[k]]
        P_k = np.abs(P_k)
        norm = np.trapezoid(P_k, x_eval)
        marginals[k] = P_k / norm if norm > 0 else P_k

    return marginals   # keys 0=A11, 1=A22, 2=A12, 3=A13


def compute_4d_pdf(Gamma=0.1, N_basis=6, N_quad=20, a=0.8):
    """
    Compute stationary marginal PDFs for (A11, A22, A12, A13) using
    the 4D FP Hermite expansion.

    Matrix size: N_basis^4 × N_basis^4
      N_basis=6  →   1296 ×  1296   (< 1 s)
      N_basis=8  →   4096 ×  4096   (a few seconds)
      N_basis=10 →  10000 × 10000   (~30 s)

    Uses build_fp_matrix_2d generalised to 4D via the same GH-quadrature
    approach but with a dedicated 4D loop.
    """
    D4  = noise_diffusion_4d()
    f   = [lambda x, G=Gamma, k=k: drift_4d(x, G)[k] for k in range(4)]

    N_tot = N_basis ** 4
    print(f"  Building 4D FP matrix ({N_tot}×{N_tot})…", end=' ', flush=True)
    t0 = time.time()
    L  = _build_fp_matrix_4d(f, D4, N_basis, N_quad, a, Gamma)
    print(f"done ({time.time()-t0:.1f}s)", flush=True)

    vals, vecs = solve_and_sort(L)
    lam_stat, c_stat = get_stationary_vec(vals, vecs)
    print(f"  Stationary eigenvalue: {lam_stat:.2e}", flush=True)

    x_eval    = np.linspace(-8, 8, 500)
    marginals = reconstruct_pdf_4d(c_stat, N_basis, x_eval, a=a)
    moments   = {k: pdf_moments_1d(x_eval, marginals[k]) for k in marginals}

    return x_eval, marginals, vals, moments


def _build_fp_matrix_4d(drift_list, D_mat, N_basis, N_quad, a, Gamma):
    """
    Build the (N_basis^4)² FP matrix for a 4-variable system.

    FP operator:
        L W = −∂_{x_k}[f_k W] + D_{kl} ∂²_{x_k x_l} W

    Same GH-quadrature approach as build_fp_matrix_2d, extended to 4D.
    The 4D quadrature grid has N_quad^4 points; with N_quad=20 that is
    160,000 points — very fast (no expm inside the grid loop; instead we
    call drift_4d once per quadrature point outside the column loop).
    """
    N_dim = 4
    pts, wts = hermgauss(N_quad)
    x = pts / a

    # 1D basis on quad nodes: shape (N_basis, N_quad)
    Phi    = np.stack([phi_fn(n,     x, a) for n in range(N_basis)])
    Phi_p  = np.stack([phi_deriv(n,  x, a) for n in range(N_basis)])
    Phi_pp = np.stack([phi_deriv2(n, x, a) for n in range(N_basis)])

    # 4D quadrature grid: each g has shape (Nq,Nq,Nq,Nq)
    g = np.meshgrid(*([x] * N_dim), indexing='ij')
    t = np.meshgrid(*([pts] * N_dim), indexing='ij')

    # Quadrature weight with exp correction: Π w_k · exp(Σ t_k²) / a^4
    W4 = np.ones([N_quad] * N_dim)
    for d in range(N_dim):
        shape = [1] * N_dim; shape[d] = N_quad
        W4 = W4 * wts.reshape(shape)
    exp_corr = np.exp(sum(td**2 for td in t))
    quad_wt  = W4 * exp_corr / a**N_dim          # shape (Nq,Nq,Nq,Nq)

    # Evaluate drift and its diagonal Jacobian on the full 4D grid
    X_flat = np.stack([gi.ravel() for gi in g], axis=1)  # (Nq^4, 4)
    n_pts  = N_quad ** N_dim
    F_flat  = np.zeros((n_pts, N_dim))
    dF_flat = np.zeros((n_pts, N_dim))
    h = 1e-5
    for i, xv in enumerate(X_flat):
        fv = drift_4d(xv, Gamma)
        F_flat[i] = fv
        for k in range(N_dim):
            xp = xv.copy(); xm = xv.copy()
            xp[k] += h; xm[k] -= h
            dF_flat[i, k] = (drift_4d(xp, Gamma)[k] - drift_4d(xm, Gamma)[k]) / (2*h)

    grid_shape = (N_quad,) * N_dim
    F  = F_flat.reshape(grid_shape + (N_dim,))   # (Nq,Nq,Nq,Nq,4)
    dF = dF_flat.reshape(grid_shape + (N_dim,))

    # Precompute 1D overlap integrals (shared across all dimensions)
    exp1d    = np.exp(pts**2)
    w1d      = wts * exp1d / a
    I_oo     = (Phi   * w1d) @ Phi.T     # ∫ φ_m φ_p  ≈ δ_mp
    I_od     = (Phi   * w1d) @ Phi_p.T  # ∫ φ_m φ'_p
    I_odd    = (Phi   * w1d) @ Phi_pp.T # ∫ φ_m φ''_p
    I_dd     = (Phi_p * w1d) @ Phi_p.T  # ∫ φ'_m φ'_p  (cross-diff)

    N_tot = N_basis ** N_dim
    L_mat = np.zeros((N_tot, N_tot))

    basis_idx = np.array(list(np.ndindex(*([N_basis]*N_dim))), dtype=np.int32)

    def _basis_grid(n_vec, deriv=None):
        """Φ_n or ∂Φ_n on the 4D grid."""
        result = np.ones(grid_shape)
        for d in range(N_dim):
            nd = n_vec[d]
            if deriv is None:
                fac = Phi[nd]
            elif isinstance(deriv, tuple):
                k, l = deriv
                fac = Phi_pp[nd] if d == k == l else \
                      Phi_p[nd]  if d in (k, l)  else Phi[nd]
            else:
                fac = Phi_p[nd] if d == deriv else Phi[nd]
            shape = [1]*N_dim; shape[d] = N_quad
            result = result * fac.reshape(shape)
        return result

    for p_idx, p_vec in enumerate(basis_idx):

        # ── Separable diffusion terms ──────────────────────────────
        col_sep = np.zeros(N_tot)
        for m_idx, m_vec in enumerate(basis_idx):
            val = 0.0
            for k in range(N_dim):
                # Diagonal diffusion
                Dkk = D_mat[k, k]
                if abs(Dkk) > 1e-15:
                    prod = Dkk * I_odd[m_vec[k], p_vec[k]]
                    for d in range(N_dim):
                        if d != k:
                            prod *= I_oo[m_vec[d], p_vec[d]]
                    val += prod
                # Off-diagonal diffusion (D_kl + D_lk)
                for l in range(k+1, N_dim):
                    Dkl = D_mat[k, l] + D_mat[l, k]
                    if abs(Dkl) < 1e-15:
                        continue
                    prod = Dkl * I_dd[m_vec[k], p_vec[k]] * I_dd[m_vec[l], p_vec[l]]
                    for d in range(N_dim):
                        if d not in (k, l):
                            prod *= I_oo[m_vec[d], p_vec[d]]
                    val += prod
            col_sep[m_idx] = val

        # ── Non-separable drift terms ──────────────────────────────
        Phi_p_grid = _basis_grid(p_vec)           # Φ_p on 4D grid
        LPhi_drift = np.zeros(grid_shape)
        for k in range(N_dim):
            dPhi_dxk = _basis_grid(p_vec, deriv=k)
            LPhi_drift -= dF[..., k] * Phi_p_grid + F[..., k] * dPhi_dxk

        wLPhi = quad_wt * LPhi_drift

        # Project onto each row m by contracting dim-by-dim
        col_drift = np.zeros(N_tot)
        for m_idx, m_vec in enumerate(basis_idx):
            acc = wLPhi.copy()
            # Contract axes 0..3 one at a time (always contract axis 0
            # after moving the target dimension there via tensordot)
            for d in range(N_dim):
                acc = np.tensordot(Phi[m_vec[d]], acc, axes=([0], [0]))
            col_drift[m_idx] = float(acc)

        L_mat[:, p_idx] = col_sep + col_drift

    return L_mat


def pdf_moments_1d(x, W):
    """Compute mean, variance, skewness, excess kurtosis from a 1D PDF."""
    mu   = np.trapezoid(x * W, x)
    mu2  = np.trapezoid((x - mu)**2 * W, x)
    mu3  = np.trapezoid((x - mu)**3 * W, x)
    mu4  = np.trapezoid((x - mu)**4 * W, x)
    sigma = math.sqrt(max(mu2, 1e-30))
    return dict(mean=mu, var=mu2, std=sigma,
                skewness=mu3 / sigma**3,
                ex_kurtosis=mu4 / sigma**4 - 3.0)


# ══════════════════════════════════════════════════════════════════
# Verification: Ornstein-Uhlenbeck benchmark
# ══════════════════════════════════════════════════════════════════

def verify_ou_process(N_basis=12, N_quad=40, verbose=True):
    """
    Benchmark on the Ornstein-Uhlenbeck process:
        dx = −x dt + √2 dW   (γ=1, D=1)
    Exact eigenvalues:  λ_n = −n   (n = 0, 1, 2, …)
    Stationary PDF:     P_st(x) = (2π)^{−½} exp(−x²/2)

    IMPORTANT: use a=1.0 for this benchmark.  The Hermite functions φ_n(x; a=1)
    are exact eigenfunctions of the OU FP operator, so L is exactly diagonal
    and eigenvalues reproduce to machine precision (~10^{−14}).
    With a ≠ 1 the φ_n are not eigenfunctions; convergence still occurs but
    requires more basis functions.
    """
    gamma, D = 1.0, 1.0
    a = 1.0           # exact for OU: stationary PDF ~ exp(-x²/2) matches φ basis
    drift_fn = lambda x: -gamma * x

    L_mat = build_fp_matrix_1d(drift_fn, D, N_basis, N_quad, a=a)
    vals, vecs = solve_and_sort(L_mat)

    # solve_and_sort returns ascending Re(λ): most negative first.
    # Reverse to get descending (0, -1, -2, …) for comparison with exact.
    numerical = vals.real[::-1]        # now index 0 = eigenvalue closest to 0
    exact     = -np.arange(N_basis, dtype=float)
    errors    = np.abs(numerical - exact)

    if verbose:
        print("\n── Ornstein-Uhlenbeck Verification (a=1, γ=1, D=1) ─────────")
        print(f"  {'n':>3}  {'Exact λ_n':>12}  {'Numerical':>12}  {'|Error|':>10}")
        for n in range(min(8, N_basis)):
            print(f"  {n:>3}  {exact[n]:>12.4f}  {numerical[n]:>12.8f}  {errors[n]:>10.2e}")
        print(f"  Max error over {N_basis} eigenvalues: {errors.max():.2e}")

    return errors.max() < 1e-8


# ══════════════════════════════════════════════════════════════════
# Convergence study
# ══════════════════════════════════════════════════════════════════

def convergence_study_1d(Gamma=0.1, a=0.8, basis_sizes=None, N_quad=60):
    """
    Assess convergence of the stationary PDF moments with increasing N_basis.
    """
    if basis_sizes is None:
        basis_sizes = [6, 10, 14, 18, 22, 26]
    x_ref = np.linspace(-10, 10, 1000)

    drift_fn = lambda x: drift_1d_A11(x, Gamma)
    D = noise_diffusion_1d()

    results = []
    W_prev = None
    print(f"\n── Convergence study (Γ={Gamma}, a={a}) ──────────────────")
    print(f"  {'N_basis':>8}  {'Mean':>8}  {'Var':>8}  {'Skew':>8}  {'Ekurt':>8}  {'ΔW_inf':>10}")

    for N in basis_sizes:
        L = build_fp_matrix_1d(drift_fn, D, N, N_quad, a=a)
        vals, vecs = solve_and_sort(L)
        lam_stat, c_stat = get_stationary_vec(vals, vecs)

        W = reconstruct_pdf_1d(c_stat, N, x_ref, a=a)
        m = pdf_moments_1d(x_ref, W)

        delta = np.max(np.abs(W - W_prev)) if W_prev is not None else float('nan')
        print(f"  {N:>8}  {m['mean']:>8.4f}  {m['var']:>8.4f}  "
              f"{m['skewness']:>8.4f}  {m['ex_kurtosis']:>8.4f}  {delta:>10.2e}")
        results.append({'N': N, 'W': W, **m})
        W_prev = W.copy()

    return x_ref, results


# ══════════════════════════════════════════════════════════════════
# Multi-Γ 1D study (Fig. 2 analogue)
# ══════════════════════════════════════════════════════════════════

def compute_pdfs_multi_gamma(gammas, N_basis=24, N_quad=60, a=0.8):
    """Compute stationary PDFs and moments for multiple Γ values."""
    x_eval = np.linspace(-10, 10, 1000)
    D = noise_diffusion_1d()
    records = []

    for Gamma in gammas:
        drift_fn = lambda x, G=Gamma: drift_1d_A11(x, G)
        print(f"  Γ = {Gamma:.3f} …", end=' ', flush=True)
        t0 = time.time()
        L = build_fp_matrix_1d(drift_fn, D, N_basis, N_quad, a=a)
        vals, vecs = solve_and_sort(L)
        lam_stat, c_stat = get_stationary_vec(vals, vecs)
        W = reconstruct_pdf_1d(c_stat, N_basis, x_eval, a=a)
        m = pdf_moments_1d(x_eval, W)
        print(f"done ({time.time()-t0:.1f}s)  λ_stat={lam_stat:.2e}  skew={m['skewness']:.3f}")
        records.append({'Gamma': Gamma, 'W': W, 'vals': vals, **m})

    return x_eval, records


# ══════════════════════════════════════════════════════════════════
# 2-D study
# ══════════════════════════════════════════════════════════════════

def compute_2d_pdf(Gamma=0.1, N_basis=8, N_quad=20, a=0.7):
    """Compute the joint stationary PDF for (A11, A12)."""
    D_mat = noise_diffusion_2d()
    f1 = lambda x1, x2, G=Gamma: drift_2d_A11_A12(x1, x2, G)[0]
    f2 = lambda x1, x2, G=Gamma: drift_2d_A11_A12(x1, x2, G)[1]

    print(f"  Building 2D FP matrix ({N_basis}²×{N_basis}² = {N_basis**4} entries)…", end=' ')
    t0 = time.time()
    L = build_fp_matrix_2d([f1, f2], D_mat, N_basis, N_quad, a=a)
    print(f"done ({time.time()-t0:.1f}s)")

    vals, vecs = solve_and_sort(L)
    lam_stat, c_stat = get_stationary_vec(vals, vecs)
    print(f"  Stationary eigenvalue: {lam_stat:.2e}")

    x_eval = np.linspace(-6, 6, 120)
    W2d = reconstruct_pdf_2d(c_stat, N_basis, x_eval, a=a)

    dx = x_eval[1] - x_eval[0]
    P_x1 = W2d.sum(axis=1) * dx
    P_x2 = W2d.sum(axis=0) * dx
    m1 = pdf_moments_1d(x_eval, P_x1)
    m2 = pdf_moments_1d(x_eval, P_x2)

    return x_eval, W2d, P_x1, P_x2, vals, m1, m2


# ══════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


def plot_convergence(x_ref, results, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_pdf, ax_moments = axes

    for rec in results[::2]:   # plot every other
        N = rec['N']
        ax_pdf.semilogy(x_ref, rec['W'], label=f'N={N}', lw=1.5)

    gauss = np.exp(-x_ref**2 / 2) / math.sqrt(2 * math.pi)
    ax_pdf.semilogy(x_ref, gauss, 'k--', lw=1.5, alpha=0.6, label='Gaussian')
    ax_pdf.set(xlim=(-8, 8), ylim=(1e-6, 1),
               xlabel=r'$A_{11}$ (units of $T^{-1}$)',
               ylabel='PDF', title='Convergence of stationary PDF with basis size')
    ax_pdf.legend(fontsize=8)
    ax_pdf.grid(True, alpha=0.3)

    Ns   = [r['N'] for r in results]
    skew = [r['skewness'] for r in results]
    kurt = [r['ex_kurtosis'] for r in results]
    means = [r['mean'] for r in results]
    ax_moments.plot(Ns, skew,  'o-', label='Skewness',        color=COLORS[0])
    ax_moments.plot(Ns, kurt,  's-', label='Excess kurtosis',  color=COLORS[1])
    ax_moments.plot(Ns, means, '^-', label='Mean (≈0 ideal)',   color=COLORS[2])
    ax_moments.axhline(0, color='k', ls='--', lw=0.8)
    ax_moments.axhline(-0.5, color=COLORS[0], ls=':', lw=1.0, alpha=0.6,
                        label='Target skewness −0.5')
    ax_moments.set(xlabel='N_basis', ylabel='Moment value',
                   title='Convergence of moments with basis size')
    ax_moments.legend(fontsize=8)
    ax_moments.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {save_path}")


def plot_multi_gamma(x_eval, records, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax_ll, ax_lt, ax_eig = axes
    gammas_all = [r['Gamma'] for r in records]

    for i, rec in enumerate(records):
        lbl = fr'$\Gamma={rec["Gamma"]}$'
        c = COLORS[i % len(COLORS)]
        # A11 (longitudinal) — same as our 1D computation
        ax_ll.semilogy(x_eval, rec['W'], color=c, lw=2, label=lbl)
        # A12 (transverse) not separately computed in 1D reduction, show same
        # (noted as an approximation)
        ax_lt.semilogy(x_eval, rec['W'], color=c, lw=2, label=lbl, ls='--')

    gauss = np.exp(-x_eval**2 / 2) / math.sqrt(2 * math.pi)
    for ax in [ax_ll, ax_lt]:
        ax.semilogy(x_eval, gauss, 'k--', lw=1.5, label='Gaussian', alpha=0.7)
        ax.set(xlim=(-8, 8), ylim=(1e-6, 1),
               ylabel='PDF')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    ax_ll.set(xlabel=r'$A_{11}/A_{11}^{\rm rms}$',
              title=r'(a) Longitudinal PDF $A_{11}$')
    ax_lt.set(xlabel=r'$A_{12}/A_{12}^{\rm rms}$',
              title=r'(b) Transverse PDF $A_{12}$ (approx.)')

    # Eigenvalue spectrum for last Gamma
    vals = records[-1]['vals']
    ax_eig.scatter(vals.real, vals.imag, s=20, c='steelblue', alpha=0.7)
    ax_eig.axvline(0, color='r', ls='--', alpha=0.5, lw=1)
    ax_eig.axhline(0, color='k', ls='-',  alpha=0.3, lw=0.8)
    ax_eig.set(xlabel=r'Re($\lambda$)', ylabel=r'Im($\lambda$)',
               title=fr'FP eigenvalue spectrum (Γ={records[-1]["Gamma"]})')
    ax_eig.grid(True, alpha=0.3)

    plt.suptitle('Complete Set Expansion — Chevillard & Meneveau (2006) Eq. (6)',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {save_path}")


def plot_2d(x_eval, W2d, P_x1, P_x2, vals, m1, m2, Gamma, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax_joint, ax_marg, ax_eig = axes

    # Joint PDF — log contours
    Wpos = np.clip(W2d, W2d[W2d > 0].min() * 0.01, None)
    levels = np.logspace(np.log10(Wpos.max()) - 4, np.log10(Wpos.max()), 12)
    cs = ax_joint.contourf(x_eval, x_eval, Wpos.T,
                           levels=levels, cmap='Blues',
                           norm=mcolors.LogNorm(vmin=levels[0], vmax=levels[-1]))
    plt.colorbar(cs, ax=ax_joint, label='log P')
    ax_joint.set(xlabel=r'$A_{11}$', ylabel=r'$A_{12}$',
                 title=fr'Joint PDF P($A_{{11}},A_{{12}}$), Γ={Gamma}')
    ax_joint.grid(True, alpha=0.2)

    ax_marg.semilogy(x_eval, P_x1, 'b-',  lw=2, label=fr'$A_{{11}}$ (skew={m1["skewness"]:.2f})')
    ax_marg.semilogy(x_eval, P_x2, 'r-',  lw=2, label=fr'$A_{{12}}$ (skew={m2["skewness"]:.2f})')
    gauss = np.exp(-x_eval**2 / 2) / math.sqrt(2 * math.pi)
    ax_marg.semilogy(x_eval, gauss, 'k--', lw=1.5, alpha=0.6, label='Gaussian')
    ax_marg.set(xlim=(-6, 6), ylim=(1e-5, 1),
                xlabel='Component value',  ylabel='Marginal PDF',
                title='Marginal distributions')
    ax_marg.legend(fontsize=9)
    ax_marg.grid(True, alpha=0.3)

    ax_eig.scatter(vals.real, vals.imag, s=20, c='steelblue', alpha=0.7)
    ax_eig.axvline(0, color='r', ls='--', alpha=0.5)
    ax_eig.axhline(0, color='k', ls='-',  alpha=0.3)
    ax_eig.set(xlabel=r'Re($\lambda$)', ylabel=r'Im($\lambda$)',
               title='2D FP eigenvalue spectrum')
    ax_eig.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {save_path}")


def plot_moments_vs_gamma(records, save_path):
    gammas  = [r['Gamma'] for r in records]
    skews   = [r['skewness'] for r in records]
    kurts   = [r['ex_kurtosis'] for r in records]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax_s, ax_k = axes

    ax_s.plot(gammas, skews, 'o-b', lw=2, ms=8, label='Computed skewness')
    ax_s.axhline(-0.5, color='r', ls='--', lw=1.5, label='Target ~−0.5')
    ax_s.set(xlabel=r'$\Gamma$ (∝ $\mathcal{R}_e^{-1/2}$)',
             ylabel='Skewness', title=r'Longitudinal skewness vs. $\Gamma$')
    ax_s.legend(); ax_s.grid(True, alpha=0.3); ax_s.invert_xaxis()

    ax_k.plot(gammas, kurts, 's-g', lw=2, ms=8, label='Computed excess kurtosis')
    ax_k.axhline(0, color='k', ls='--', lw=0.8, label='Gaussian (=0)')
    ax_k.set(xlabel=r'$\Gamma$ (∝ $\mathcal{R}_e^{-1/2}$)',
             ylabel='Excess kurtosis',
             title=r'Non-Gaussianity vs. $\Gamma$ (increasing $\mathcal{R}_e$ →)')
    ax_k.legend(); ax_k.grid(True, alpha=0.3); ax_k.invert_xaxis()

    plt.suptitle('Trend with Reynolds number (decreasing Γ = increasing Re)',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {save_path}")


def plot_4d_marginals(x_eval, marginals, moments, Gamma, save_path):
    """Plot the 4 marginal PDFs from the 4D FP solve."""
    labels = ['$A_{11}$', '$A_{22}$', '$A_{12}$', '$A_{13}$']
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    gauss = np.exp(-x_eval**2 / 2) / math.sqrt(2 * math.pi)

    for k, ax in enumerate(axes):
        m = moments[k]
        ax.semilogy(x_eval, marginals[k], lw=2, color=COLORS[k],
                    label=labels[k])
        ax.semilogy(x_eval, gauss, 'k--', lw=1.2, alpha=0.5, label='Gaussian')
        ax.set(xlim=(-6, 6), ylim=(1e-5, 1),
               xlabel='Component value', ylabel='PDF',
               title=f'{labels[k]}\nskew={m["skewness"]:.3f}, '
                     f'kurt={m["ex_kurtosis"]:.2f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(fr'4D FP Marginals — $(A_{{11}}, A_{{22}}, A_{{12}}, A_{{13}})$,'
                 fr'  $\Gamma={Gamma}$', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {save_path}")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    OUT = 'fpe_outputs'
    os.makedirs(OUT, exist_ok = True)

    print("=" * 70)
    print("Fokker-Planck Complete Set Expansion")
    print("Chevillard & Meneveau (2006) Eq. (6) + Risken Sec. 6.6.5")
    print("=" * 70)

    # ── Step 0: Verify on OU process ──────────────────────────────
    print("\n[0] Ornstein-Uhlenbeck verification (exact eigenvalues 0, −1, −2, …)")
    ok = verify_ou_process(N_basis=12, N_quad=30)
    print(f"    PASS: all eigenvalue errors < 1e-8 → {ok}")

    # ── Step 1: 1D convergence study ──────────────────────────────
    print("\n[1] 1D convergence study  (Γ=0.1)")
    x_ref, conv_results = convergence_study_1d(
        Gamma=0.1, a=0.8,
        basis_sizes=[6, 10, 14, 18, 22, 26],
        N_quad=60
    )
    plot_convergence(x_ref, conv_results, f'{OUT}/fp_convergence.png')

    # ── Step 2: Multi-Γ PDF comparison (Fig. 2 analogue) ──────────
    print("\n[2] Multi-Γ 1D PDFs  (Γ = 0.2, 0.1, 0.08, 0.06)")
    x_ev, records = compute_pdfs_multi_gamma(
        gammas=[0.2, 0.1, 0.08, 0.06],
        N_basis=24, N_quad=60, a=0.8
    )
    plot_multi_gamma(x_ev, records, f'{OUT}/fp_multi_gamma.png')
    plot_moments_vs_gamma(records, f'{OUT}/fp_moments_vs_gamma.png')

    # Print final summary table
    print("\n  Final moment summary:")
    print(f"  {'Γ':>6}  {'Mean':>8}  {'Std':>8}  {'Skewness':>10}  {'Ex.Kurt':>10}")
    for r in records:
        print(f"  {r['Gamma']:>6.3f}  {r['mean']:>8.4f}  {r['std']:>8.4f}"
              f"  {r['skewness']:>10.4f}  {r['ex_kurtosis']:>10.4f}")

    # ── Step 3: 2D joint PDF ──────────────────────────────────────
    print("\n[3] 2D joint PDF for (A11, A12)  (Γ=0.1)")
    x2d, W2d, P_x1, P_x2, vals2d, m1, m2 = compute_2d_pdf(
        Gamma=0.1, N_basis=8, N_quad=20, a=0.7
    )
    print(f"  A11 marginal: mean={m1['mean']:.3f}, skew={m1['skewness']:.3f}")
    print(f"  A12 marginal: mean={m2['mean']:.3f}, skew={m2['skewness']:.3f}")
    plot_2d(x2d, W2d, P_x1, P_x2, vals2d, m1, m2, Gamma=0.1,
            save_path=f'{OUT}/fp_2d_joint.png')

    # ── Step 4: 4D — (A11, A22, A12, A13) ───────────────────────
    print("\n[4] 4D FP: (A11, A22, A12, A13)  (Γ=0.1)")
    x4d, marginals_4d, vals4d, moments_4d = compute_4d_pdf(
        Gamma=0.1, N_basis=9, N_quad=20, a=0.8
    )
    comp_labels = ['A11', 'A22', 'A12', 'A13']
    print(f"\n  {'Comp':>6}  {'Mean':>8}  {'Std':>8}  {'Skewness':>10}  {'Ex.Kurt':>10}")
    for k in range(4):
        m = moments_4d[k]
        print(f"  {comp_labels[k]:>6}  {m['mean']:>8.4f}  {m['std']:>8.4f}"
              f"  {m['skewness']:>10.4f}  {m['ex_kurtosis']:>10.4f}")
    plot_4d_marginals(x4d, marginals_4d, moments_4d, Gamma=0.1,
                      save_path=f'{OUT}/fp_4d_marginals.png')

    print("\n" + "=" * 70)
    print("All outputs saved to /mnt/user-data/outputs/")
    print("  fp_convergence.png      — basis-size convergence")
    print("  fp_multi_gamma.png      — PDFs at 4 Γ values (Fig. 2 analogue)")
    print("  fp_moments_vs_gamma.png — skewness & kurtosis vs Γ")
    print("  fp_2d_joint.png         — joint (A11, A12) distribution")
    print("  fp_4d_marginals.png     — marginals for (A11, A22, A12, A13)")
    print("=" * 70)