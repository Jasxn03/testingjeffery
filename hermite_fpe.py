"""
Complete-set Hermite expansion for the Fokker-Planck equation of
Chevillard & Meneveau (2006), Eq. (6) — full 8-dimensional treatment.

SDE (Eq. 6):
    dA = [ -A² + Tr(A²)/Tr(C⁻¹) C⁻¹ - Tr(C⁻¹)/3 · A ] dt + dW
    C_Γ = exp(Γ·A) exp(Γ·Aᵀ)

State space: 8 independent components x = [A₀₀, A₀₁, A₀₂, A₁₀, A₁₁, A₁₂, A₂₀, A₂₁]
             with A₂₂ = -A₀₀ - A₁₁ (tracelessness).

Method: Risken §6.6.5 — expand W = Σ_n c_n(t) Φ_n(y)
    Φ_n(y) = ∏_α φ_{n_α}(y_α; a_α)   (tensor-product Hermite functions)
    y = Vᵀ x  (coordinate rotation that diagonalises D)
    a_α = 1/√Λ_α  (scale matched to stationary OU variance in each principal direction)

FPE:  ∂W/∂t = L_FP W = -∂/∂x_α[D_α(x) W] + D_αβ ∂²W/∂x_α∂x_β

Matrix element (integration-by-parts form, avoids divergence of drift):
    L[m,n] = Σ_α ∫ (∂Φ_m/∂x_α) D_α(x) Φ_n(x) d⁸x
           + Σ_{α,β} D_αβ ∫ Φ_m(x) ∂²Φ_n/∂x_α∂x_β d⁸x

Quadrature: Gauss-Hermite with bare-Hermite (ψ) form avoids numerical
overflow from exp(+Σ t²) in 8 dimensions.
"""

import math
import time
import itertools
import warnings
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.linalg import eig, expm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ─────────────────────────────────────────────────────────────────
# 1.  PROBLEM GEOMETRY
# ─────────────────────────────────────────────────────────────────

NDIM      = 8                             # independent components of traceless A
IDX_TO_IJ = [(0,0),(0,1),(0,2),           # mapping: component index → (i,j) of A
              (1,0),(1,1),(1,2),
              (2,0),(2,1)]

# Noise diffusion tensor D_αβ = ½ <dW_α dW_β>/dt = <G_{iα,jα} G_{iβ,jβ}>
# from <G_ij G_kl> = 2δ_ik δ_jl - ½δ_ij δ_kl - ½δ_il δ_jk
def _build_diffusion_matrix():
    D = np.zeros((NDIM, NDIM))
    for a, (ia, ja) in enumerate(IDX_TO_IJ):
        for b, (ib, jb) in enumerate(IDX_TO_IJ):
            D[a, b] = (2*(ia==ib)*(ja==jb)
                       - 0.5*(ia==ja)*(ib==jb)
                       - 0.5*(ia==jb)*(ja==ib))
    return D

D_MAT  = _build_diffusion_matrix()           # 8×8 constant diffusion matrix
# Diagonalise: D = V Λ Vᵀ → in y = Vᵀ x coords diffusion is diagonal
_Lam, _V = np.linalg.eigh(D_MAT)
D_EIGVALS = _Lam                             # (8,) eigenvalues of D_MAT
D_EIGVECS = _V                               # (8,8) columns = eigenvectors


# ─────────────────────────────────────────────────────────────────
# 2.  COORDINATE MAPS
# ─────────────────────────────────────────────────────────────────

def x_to_A(x):
    """8-vector x → 3×3 traceless matrix A."""
    A = np.zeros((3, 3))
    for alpha, (i, j) in enumerate(IDX_TO_IJ):
        A[i, j] = x[alpha]
    A[2, 2] = -x[0] - x[4]
    return A

def y_to_x(y):
    """Principal coordinates y → original components x = V y."""
    return D_EIGVECS @ y

def x_to_y(x):
    """Original components x → principal coordinates y = Vᵀ x."""
    return D_EIGVECS.T @ x


# ─────────────────────────────────────────────────────────────────
# 3.  CHEVILLARD-MENEVEAU DRIFT  (Eq. 6)
# ─────────────────────────────────────────────────────────────────

def cm_drift_x(x, Gamma):
    """
    Drift vector in original x-coordinates.
    Returns 8-vector D_α(x) = (bracket of Eq.6)_α.
    """
    A    = x_to_A(x)
    eGA  = expm(Gamma * A)
    C    = eGA @ eGA.T
    Cinv = np.linalg.inv(C)
    trA2   = np.trace(A @ A)
    trCinv = np.trace(Cinv)
    Dm = -A @ A + (trA2 / trCinv) * Cinv - (trCinv / 3.0) * A
    return np.array([Dm[i, j] for (i, j) in IDX_TO_IJ])

def cm_drift_y(y, Gamma):
    """Drift in principal y-coordinates: f_y(y) = Vᵀ f_x(V y)."""
    x = y_to_x(y)
    return D_EIGVECS.T @ cm_drift_x(x, Gamma)


# ─────────────────────────────────────────────────────────────────
# 4.  HERMITE FUNCTIONS (bare ψ form, evaluated at GH nodes)
# ─────────────────────────────────────────────────────────────────
#
# φ_n(x; a) = √a · N_n · H_n(ax) · exp(-a²x²/2)   [standard normalised form]
#
# At GH node t = ax:
# ψ_n(t) = N_n · H_n(t)                 [bare; no Gaussian; ∫ψ_m ψ_n e^{-t²}dt = δ_mn]
# ψ'_n(t, a) = a · N_n · [2n H_{n-1} − t H_n]  [= dφ_n/dx stripped of √a·exp]
# ψ''_n(t, a) = a² · N_n · [(t²−1)H_n − 4nt H_{n-1} + 4n(n−1) H_{n-2}]
#
# GH integration without Gaussian overflow:
# ∫ Φ_m(x) g(x) Φ_n(x) d⁸x = Σ_K (∏ w_{k_α}) · ψ_m(t_K) · g(t_K/a) · ψ_n(t_K)
# (no exp correction needed — GH weights already carry exp(-Σt²))

def _hermite_poly(n, t):
    """Physicist's Hermite polynomial H_n(t) via three-term recurrence."""
    t = np.asarray(t, float)
    if n == 0:
        return np.ones_like(t)
    if n == 1:
        return 2.0 * t
    Hm2, Hm1 = np.ones_like(t), 2.0 * t
    for k in range(2, n + 1):
        H = 2.0 * t * Hm1 - 2.0 * (k - 1) * Hm2
        Hm2, Hm1 = Hm1, H
    return Hm1

def _hermite_norm(n):
    return 1.0 / math.sqrt(2**n * math.factorial(n) * math.sqrt(math.pi))

def psi(n, t):
    """ψ_n(t) = N_n H_n(t)."""
    return _hermite_norm(n) * _hermite_poly(n, t)

def dpsi(n, t, a):
    """ψ'_n(t, a) = a N_n [2n H_{n-1}(t) − t H_n(t)]  (= dφ_n/dx at x=t/a, stripped)."""
    Nn  = _hermite_norm(n)
    Hn  = _hermite_poly(n, t)
    val = -t * Hn
    if n >= 1:
        val = val + 2 * n * _hermite_poly(n - 1, t)
    return a * Nn * val

def d2psi(n, t, a):
    """ψ''_n(t, a) = a² N_n [(t²−1)H_n − 4nt H_{n-1} + 4n(n−1) H_{n-2}]."""
    Nn  = _hermite_norm(n)
    Hn  = _hermite_poly(n, t)
    val = (t**2 - 1.0) * Hn
    if n >= 1:
        val = val - 4.0 * n * t * _hermite_poly(n - 1, t)
    if n >= 2:
        val = val + 4.0 * n * (n - 1) * _hermite_poly(n - 2, t)
    return a**2 * Nn * val


# ─────────────────────────────────────────────────────────────────
# 5.  BASIS MULTI-INDICES
# ─────────────────────────────────────────────────────────────────

def build_basis(n_max):
    """All 8-tuples n with Σ n_α ≤ n_max (restricted total-order basis)."""
    return [b for b in itertools.product(*[range(n_max + 1)] * NDIM)
            if sum(b) <= n_max]


# ─────────────────────────────────────────────────────────────────
# 6.  FPE MATRIX CONSTRUCTION
# ─────────────────────────────────────────────────────────────────

def build_fpe_matrix(Gamma, n_max, n_quad, verbose=True):
    """
    Build the FPE operator matrix L (N_basis × N_basis) such that ċ = Lc.

    Uses the diagonalised diffusion coordinate system y = Vᵀx:
      • In y-coords the diffusion is diagonal: D_y = diag(Λ).
      • Hermite scale per principal direction: a_α = 1/√Λ_α.
      • Drift in y-coords: f_y(y) = Vᵀ f_x(V y).

    Parameters
    ----------
    Gamma   : float   — CM parameter (∝ Re^{-1/2})
    n_max   : int     — max total Hermite order
    n_quad  : int     — GH quadrature points per dimension
    verbose : bool

    Returns
    -------
    L       : (N_basis, N_basis) array
    basis   : list of N_basis multi-index tuples
    """
    basis  = build_basis(n_max)
    N_bas  = len(basis)
    a_vec  = 1.0 / np.sqrt(D_EIGVALS)          # scale per principal direction
    D_diag = D_EIGVALS                          # diagonal of D in y-coords

    if verbose:
        print(f"  Basis: {N_bas} functions  (n_max={n_max})")
        print(f"  Quadrature: {n_quad}^8 = {n_quad**NDIM:,} points per dim")

    # ── GH quadrature grid ──────────────────────────────────────
    pts, wts = hermgauss(n_quad)
    k_ind    = np.array(list(itertools.product(range(n_quad), repeat=NDIM)))
    n_grid   = len(k_ind)                     # n_quad**8

    # GH node t_α at each grid point for each principal direction
    t_at_K = pts[k_ind]                       # (n_grid, 8)  — GH nodes per dim

    # Physical y-coordinates at each grid point: y_α = t_{k_α} / a_α
    y_at_K = (t_at_K / a_vec[None, :]).T      # (8, n_grid)

    # Product weights (no exp correction — using bare ψ)
    w_grid = np.prod(wts[k_ind], axis=1)      # (n_grid,)

    # ── Precompute ψ, ψ', ψ'' at GH nodes ──────────────────────
    N_phi   = n_max + 2                        # need one extra for recurrences
    psi_K   = np.zeros((NDIM, N_phi, n_grid))
    dpsi_K  = np.zeros((NDIM, N_phi, n_grid))
    d2psi_K = np.zeros((NDIM, N_phi, n_grid))
    for alpha in range(NDIM):
        a = a_vec[alpha]
        t = t_at_K[:, alpha]
        for n in range(N_phi):
            psi_K[alpha, n]   = psi(n, t)
            dpsi_K[alpha, n]  = dpsi(n, t, a)
            d2psi_K[alpha, n] = d2psi(n, t, a)

    # ── Ψ_m matrix: Ψ_m(K) = ∏_α ψ_{m_α}(t_{k_α}) ────────────
    Psi_mat = np.ones((N_bas, n_grid))
    for j, b in enumerate(basis):
        for alpha in range(NDIM):
            Psi_mat[j] *= psi_K[alpha, b[alpha]]

    # ── Drift at all grid points ─────────────────────────────────
    if verbose:
        print(f"  Computing drift at {n_grid:,} quadrature points …", flush=True)
    t0 = time.time()

    # Map y → x → drift in x → drift in y
    x_at_K = (D_EIGVECS @ y_at_K)             # x = V y,  shape (8, n_grid)
    f_y_grid = np.zeros((NDIM, n_grid))
    for K in range(n_grid):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fx = cm_drift_x(x_at_K[:, K], Gamma)
        f_y_grid[:, K] = D_EIGVECS.T @ fx     # f_y = Vᵀ f_x

    if verbose:
        print(f"    done in {time.time()-t0:.1f}s")

    # ── Assemble L ───────────────────────────────────────────────
    if verbose:
        print("  Assembling FP matrix …", flush=True)
    t0 = time.time()
    L = np.zeros((N_bas, N_bas))

    # — Drift term (integration-by-parts form) —
    # L_drift[m,n] = Σ_α Σ_K w_K DΨ_{m,α}(K) f_{y,α}(K) Ψ_n(K)
    # where DΨ_{m,α}(K) = ψ'_{m_α}(t_{k_α}) · ∏_{β≠α} ψ_{m_β}(t_{k_β})
    for alpha in range(NDIM):
        DPsi_alpha = np.ones((N_bas, n_grid))
        for j, b in enumerate(basis):
            for beta in range(NDIM):
                arr = dpsi_K[beta, b[beta]] if beta == alpha else psi_K[beta, b[beta]]
                DPsi_alpha[j] *= arr
        # L += DPsi_alpha @ diag(f_α w) @ Psi_mat.T
        L += DPsi_alpha @ (f_y_grid[alpha] * w_grid * Psi_mat).T

    # — Diffusion term (diagonal in y-coords) —
    # L_diff[m,n] = Σ_α Λ_α Σ_K w_K Ψ_m(K) ψ''_{n_α}(K) ∏_{β≠α} ψ_{n_β}(K)
    # (cross terms Λ_αβ with α≠β vanish since D_y is diagonal)
    for j, n_idx in enumerate(basis):
        Diff_col = np.zeros(n_grid)
        for alpha in range(NDIM):
            D2_col = np.ones(n_grid)
            for beta in range(NDIM):
                n_b = n_idx[beta]
                D2_col *= d2psi_K[beta, n_b] if beta == alpha else psi_K[beta, n_b]
            Diff_col += D_diag[alpha] * D2_col
        L[:, j] += Psi_mat @ (w_grid * Diff_col)

    if verbose:
        print(f"    done in {time.time()-t0:.1f}s")

    return L, basis


# ─────────────────────────────────────────────────────────────────
# 7.  EIGENVALUE ANALYSIS
# ─────────────────────────────────────────────────────────────────

def solve_eigenvalues(L):
    """
    Eigenvalues of L, sorted so the stationary state (λ ≈ 0) comes first.
    Returns eigenvalues and eigenvectors sorted by Re(λ) descending.
    """
    vals, vecs = eig(L)
    order = np.argsort(vals.real)[::-1]    # descending: 0 first, most negative last
    return vals[order], vecs[:, order]


def get_stationary(vals, vecs):
    """Return (λ_stat, c_stat) — eigenpair closest to λ=0."""
    idx = np.argmin(np.abs(vals.real))
    return vals[idx], vecs[:, idx].real


# ─────────────────────────────────────────────────────────────────
# 8.  MARGINAL PDF RECONSTRUCTION
# ─────────────────────────────────────────────────────────────────

def marginal_pdf_1d(c_stat, basis, component, x_eval, a_vec,
                    n_quad_int=50):
    """
    Compute the 1D marginal PDF of x_{component} by integrating out all others.

    ∫ Φ_n(y) dy_{others} = φ_{n_α}(y_α; a_α) · ∏_{β≠α} ∫ φ_{n_β} dy_β

    ∫ φ_n(y; a) dy = (√(2π)/a) · δ_{n,0}   (only ground state survives)
    So only basis elements with n_β=0 for all β≠component contribute.

    NOTE: component refers to the index in the original x-space. We project
    onto x_{component} by summing over the principal (y) directions.
    """
    # In y-space, the marginal over all y_β≠α is not directly a 1D marginal
    # in a single original component x_i, because y=Vᵀx mixes components.
    # For simplicity, reconstruct on a grid and marginalise numerically.
    pts_q, wts_q = hermgauss(n_quad_int)
    a_c = a_vec[component]

    # 1D integrals of φ_n (the ground-state integral in each non-target dim)
    sqrt2pi_over_a = math.sqrt(2 * math.pi)
    def int_phi0(a_beta):
        return sqrt2pi_over_a / a_beta

    W = np.zeros(len(x_eval))
    for j, b in enumerate(basis):
        if c_stat[j] == 0:
            continue
        # Check: all β≠component must have n_β=0
        other_zero = all(b[beta] == 0 for beta in range(NDIM) if beta != component)
        if not other_zero:
            continue
        # Contribution: c_j * φ_{b[component]}(y_component) * ∏_{β≠comp} ∫φ_0 dy_β
        prod_other = 1.0
        for beta in range(NDIM):
            if beta != component:
                prod_other *= int_phi0(a_vec[beta])
        n_c = b[component]
        phi_vals = (math.sqrt(a_c) * _hermite_norm(n_c)
                    * _hermite_poly(n_c, a_c * x_eval)
                    * np.exp(-0.5 * (a_c * x_eval)**2))
        W += c_stat[j] * prod_other * phi_vals

    W = np.abs(W)
    norm = np.trapezoid(W, x_eval)
    return W / max(norm, 1e-30)


def reconstruct_pdf_2d_principal(c_stat, basis, alpha1, alpha2, y_eval,
                                  a_vec):
    """
    Joint PDF of two principal coordinates (y_{alpha1}, y_{alpha2}).
    Marginalises over all other principal coordinates.
    """
    sqrt2pi = math.sqrt(2 * math.pi)
    def int_phi0(a): return sqrt2pi / a

    ny = len(y_eval)
    W = np.zeros((ny, ny))

    for j, b in enumerate(basis):
        if abs(c_stat[j]) < 1e-15:
            continue
        if not all(b[beta] == 0 for beta in range(NDIM)
                   if beta not in (alpha1, alpha2)):
            continue
        prod_other = math.prod(int_phi0(a_vec[beta])
                               for beta in range(NDIM)
                               if beta not in (alpha1, alpha2))
        n1, n2 = b[alpha1], b[alpha2]
        a1, a2 = a_vec[alpha1], a_vec[alpha2]
        phi1 = (math.sqrt(a1) * _hermite_norm(n1)
                * _hermite_poly(n1, a1 * y_eval)
                * np.exp(-0.5 * (a1 * y_eval)**2))
        phi2 = (math.sqrt(a2) * _hermite_norm(n2)
                * _hermite_poly(n2, a2 * y_eval)
                * np.exp(-0.5 * (a2 * y_eval)**2))
        W += c_stat[j] * prod_other * phi1[:, None] * phi2[None, :]

    W = np.abs(W)
    dy = y_eval[1] - y_eval[0]
    return W / max(np.sum(W) * dy**2, 1e-30)


# ─────────────────────────────────────────────────────────────────
# 9.  MOMENTS
# ─────────────────────────────────────────────────────────────────

def compute_moments(c_stat, basis, a_vec, n_quad_mom=40):
    """
    Compute <x_α>, <x_α²>, and <A_ij A_ij> from stationary coefficients.
    Uses 1D GH quadrature for each marginal integral.
    """
    pts, wts = hermgauss(n_quad_mom)
    sqrt2pi = math.sqrt(2 * math.pi)

    def int_phi0(a_beta):
        return sqrt2pi / a_beta

    def int_xp_phin(p, n, a):
        """∫ x^p φ_n(x;a) dx via 1D GH: x=t/a."""
        t = pts
        Hn = _hermite_poly(n, t)
        Nn = _hermite_norm(n)
        # φ_n(t/a; a) = √a N_n H_n(t) exp(-t²/2)
        # ∫ (t/a)^p √a N_n H_n(t) exp(-t²/2) dt/a = (1/a^p) * N_n * ∫ t^p H_n exp(-t²/2) dt
        # ∫ f(t) exp(-t²/2) dt = √(2π) E[f(Z)] where Z~N(0,1)
        # = √(2π) Σ_k w_k f(√2 pts_k) / √π  ... use change t→√2 u, exp(-t²/2)=exp(-u²)
        # Simpler: remap to Gauss-Hermite for exp(-t²):
        # ∫ t^p H_n(t) exp(-t²/2) dt — not standard GH.
        # Use: ∫ f(t) exp(-t²/2) dt = √2 · GH(f(√2 u)) with u=t/√2
        u = pts  # GH nodes for exp(-u²)
        t_vals = math.sqrt(2) * u
        Hn_vals = _hermite_poly(n, t_vals)
        x_vals = t_vals / a
        integrand = x_vals**p * Nn * Hn_vals
        return math.sqrt(2) * math.sqrt(a) * np.sum(wts * integrand)

    # <x_α> and <x_α²> in principal coordinates y, then transform back
    means_y  = np.zeros(NDIM)
    vars_y   = np.zeros(NDIM)

    for j, b in enumerate(basis):
        cj = c_stat[j]
        if abs(cj) < 1e-15:
            continue
        for alpha in range(NDIM):
            a = a_vec[alpha]
            prod_other = math.prod(
                int_phi0(a_vec[beta])
                for beta in range(NDIM) if beta != alpha
            )
            if all(b[beta] == 0 for beta in range(NDIM) if beta != alpha):
                n_a = b[alpha]
                means_y[alpha] += cj * prod_other * int_xp_phin(1, n_a, a)
                vars_y[alpha]  += cj * prod_other * int_xp_phin(2, n_a, a)

    # Transform mean and variance back to x-space (approximate; cross-terms ignored for speed)
    # <x> = V <y>
    means_x = D_EIGVECS @ means_y
    # <x²_α> ≈ Σ_β V²_{αβ} <y²_β>  (exact for diagonal cross-correlations = 0)
    vars_x = (D_EIGVECS**2) @ vars_y

    return {'mean_y': means_y, 'var_y': vars_y,
            'mean_x': means_x, 'var_x': vars_x}


# ─────────────────────────────────────────────────────────────────
# 10.  OU VERIFICATION (using the diagonal y-coord system)
# ─────────────────────────────────────────────────────────────────

def verify_ou_limit(n_max=2, n_quad=4, verbose=True):
    """
    Verify against the exact OU process in principal y-coordinates:
        dY_α = -Y_α dt + √(2Λ_α) dW_α
    FPE eigenvalues in the Hermite basis: exactly −Σ n_α.
    Uses the exact linear drift f_y(y) = -y (bypasses the CM expm call).

    With a_α = 1/√Λ_α the Hermite functions ARE the exact eigenfunctions
    of this diagonal OU system, so the L matrix should be exactly diagonal
    and eigenvalues reproduce to GH quadrature accuracy.
    """
    if verbose:
        print(f"\n── OU verification (exact linear drift, n_max={n_max}, n_quad={n_quad}) ──")

    a_vec  = 1.0 / np.sqrt(D_EIGVALS)
    basis  = build_basis(n_max)
    N_bas  = len(basis)

    pts, wts = hermgauss(n_quad)
    k_ind    = np.array(list(itertools.product(range(n_quad), repeat=NDIM)))
    n_grid   = len(k_ind)
    t_at_K   = pts[k_ind]
    y_at_K   = (t_at_K / a_vec[None, :]).T
    w_grid   = np.prod(wts[k_ind], axis=1)

    N_phi   = n_max + 2
    psi_K   = np.zeros((NDIM, N_phi, n_grid))
    dpsi_K  = np.zeros((NDIM, N_phi, n_grid))
    d2psi_K = np.zeros((NDIM, N_phi, n_grid))
    for alpha in range(NDIM):
        a = a_vec[alpha]
        t = t_at_K[:, alpha]
        for n in range(N_phi):
            psi_K[alpha, n]   = psi(n, t)
            dpsi_K[alpha, n]  = dpsi(n, t, a)
            d2psi_K[alpha, n] = d2psi(n, t, a)

    Psi_mat = np.ones((N_bas, n_grid))
    for j, b in enumerate(basis):
        for alpha in range(NDIM):
            Psi_mat[j] *= psi_K[alpha, b[alpha]]

    f_grid = -y_at_K   # exact OU drift: f_y(y) = -y

    L = np.zeros((N_bas, N_bas))
    for alpha in range(NDIM):
        DPsi_a = np.ones((N_bas, n_grid))
        for j, b in enumerate(basis):
            for beta in range(NDIM):
                DPsi_a[j] *= dpsi_K[beta, b[beta]] if beta == alpha else psi_K[beta, b[beta]]
        L += DPsi_a @ (f_grid[alpha] * w_grid * Psi_mat).T

    for j, n_idx in enumerate(basis):
        Diff = np.zeros(n_grid)
        for alpha in range(NDIM):
            D2 = np.ones(n_grid)
            for beta in range(NDIM):
                D2 *= d2psi_K[beta, n_idx[beta]] if beta == alpha else psi_K[beta, n_idx[beta]]
            Diff += D_EIGVALS[alpha] * D2
        L[:, j] += Psi_mat @ (w_grid * Diff)

    vals, _ = solve_eigenvalues(L)
    numerical = np.sort(vals.real)[::-1]
    expected  = sorted(set(-sum(b) for b in basis), reverse=True)

    errors = [abs(numerical[k] - expected[k]) for k in range(len(expected))]
    max_err = max(errors)

    if verbose:
        print(f"  {'k':>3}  {'Expected':>10}  {'Numerical':>12}  {'|Error|':>10}")
        for k in range(min(n_max + 2, len(expected))):
            print(f"  {k:>3}  {expected[k]:>10.1f}  {numerical[k]:>12.6f}  {errors[k]:>10.2e}")
        status = "PASS" if max_err < 1e-6 else "WARN"
        print(f"  Max error over {len(expected)} levels: {max_err:.2e}  [{status}]")
    return max_err


# ─────────────────────────────────────────────────────────────────
# 11.  CONVERGENCE STUDY
# ─────────────────────────────────────────────────────────────────

def convergence_study(Gamma, n_max_list, n_quad, verbose=True):
    """Run the solver for several n_max values and track the stationary eigenvalue."""
    results = []
    if verbose:
        print(f"\n── Convergence study  Γ={Gamma}  n_quad={n_quad} ──")
        print(f"  {'n_max':>6}  {'N_basis':>8}  {'λ_stat':>12}  {'Time(s)':>8}")
    for n_max in n_max_list:
        t0 = time.time()
        L, basis = build_fpe_matrix(Gamma=Gamma, n_max=n_max,
                                     n_quad=n_quad, verbose=False)
        vals, vecs = solve_eigenvalues(L)
        lam_stat, c_stat = get_stationary(vals, vecs)
        dt = time.time() - t0
        if verbose:
            print(f"  {n_max:>6}  {len(basis):>8}  {lam_stat.real:>12.4f}  {dt:>8.1f}")
        results.append(dict(n_max=n_max, N_basis=len(basis),
                            L=L, basis=basis, vals=vals, vecs=vecs,
                            lam_stat=lam_stat, c_stat=c_stat))
    return results


# ─────────────────────────────────────────────────────────────────
# 12.  MULTI-GAMMA STUDY
# ─────────────────────────────────────────────────────────────────

def multi_gamma_study(gammas, n_max, n_quad, verbose=True):
    """Solve for each Gamma and collect stationary eigenvalue + moments."""
    results = []
    for Gamma in gammas:
        if verbose:
            print(f"\n  Γ = {Gamma:.3f}:", flush=True)
        L, basis = build_fpe_matrix(Gamma=Gamma, n_max=n_max,
                                     n_quad=n_quad, verbose=verbose)
        vals, vecs = solve_eigenvalues(L)
        lam_stat, c_stat = get_stationary(vals, vecs)
        a_vec = 1.0 / np.sqrt(D_EIGVALS)
        moms  = compute_moments(c_stat, basis, a_vec)
        if verbose:
            print(f"    λ_stat = {lam_stat.real:.4e}")
            print(f"    <A₁₁> = {moms['mean_x'][0]:.4f}   <A₁₁²> = {moms['var_x'][0]:.4f}")
        results.append(dict(Gamma=Gamma, L=L, basis=basis,
                            vals=vals, lam_stat=lam_stat,
                            c_stat=c_stat, moments=moms))
    return results


# ─────────────────────────────────────────────────────────────────
# 13.  PLOTTING
# ─────────────────────────────────────────────────────────────────

def plot_eigenvalue_spectrum(results, save_path):
    """Plot FP eigenvalue spectra for all Gamma values."""
    n_gamma = len(results)
    fig, axes = plt.subplots(1, n_gamma, figsize=(4 * n_gamma, 4))
    if n_gamma == 1:
        axes = [axes]
    for ax, res in zip(axes, results):
        vals = res['vals']
        ax.scatter(vals.real, vals.imag, s=18, alpha=0.7, c='steelblue')
        ax.axvline(0, color='r', ls='--', lw=0.8, alpha=0.6)
        ax.axhline(0, color='k', ls='-',  lw=0.5, alpha=0.3)
        ax.set(xlabel='Re(λ)', ylabel='Im(λ)',
               title=f'Γ={res["Gamma"]:.3f}\nN={len(res["basis"])}')
        ax.grid(True, alpha=0.25)
    plt.suptitle('FP eigenvalue spectra — CM (2006) 8D system', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {save_path}")


def plot_convergence(results, save_path):
    """Plot stationary eigenvalue vs basis size."""
    n_max_list  = [r['n_max'] for r in results]
    n_bas_list  = [r['N_basis'] for r in results]
    lam_list    = [r['lam_stat'].real for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.semilogy(n_max_list, np.abs(lam_list), 'o-b', ms=7, lw=2)
    ax1.set(xlabel='n_max', ylabel='|λ_stationary|',
            title='Stationary eigenvalue convergence')
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(n_bas_list, np.abs(lam_list), 's-g', ms=7, lw=2)
    ax2.set(xlabel='N_basis', ylabel='|λ_stationary|',
            title='Stationary eigenvalue vs basis size')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {save_path}")


def plot_marginal_pdfs(mg_results, comp=0, save_path=None):
    """
    Plot marginal PDFs of x_{comp} for multiple Gamma values.
    comp=0 → A₁₁ (longitudinal), comp=1 → A₁₂ (transverse).
    """
    a_vec  = 1.0 / np.sqrt(D_EIGVALS)
    x_eval = np.linspace(-8, 8, 600)
    comp_names = ['A₀₀','A₀₁','A₀₂','A₁₀','A₁₁','A₁₂','A₂₀','A₂₁']

    fig, (ax_log, ax_lin) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728']

    for res, col in zip(mg_results, colors):
        lbl = f"Γ={res['Gamma']:.3f}"
        # Build marginal in principal y-coordinates, then convert axis
        # For principal direction closest to x_{comp}:
        # (approximate: use the y-direction that maximises |V_{comp,alpha}|)
        alpha_best = np.argmax(np.abs(D_EIGVECS[comp, :]))
        y_eval = x_eval  # same axis, just labelled differently
        W = marginal_pdf_1d(res['c_stat'], res['basis'], alpha_best,
                            y_eval, a_vec)
        ax_log.semilogy(x_eval, W, color=col, lw=2, label=lbl)
        ax_lin.plot(x_eval, W,    color=col, lw=2, label=lbl)

    # Gaussian reference
    gauss = np.exp(-x_eval**2 / 2) / math.sqrt(2 * math.pi)
    ax_log.semilogy(x_eval, gauss, 'k--', lw=1.5, alpha=0.6, label='Gaussian')
    ax_lin.plot(x_eval, gauss,    'k--', lw=1.5, alpha=0.6, label='Gaussian')

    for ax, yscale in [(ax_log, 'log'), (ax_lin, 'linear')]:
        ax.set(xlabel=f'{comp_names[comp]}', ylabel='PDF',
               title=f'Marginal PDF of {comp_names[comp]} ({yscale})')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    ax_log.set(ylim=(1e-6, 1))

    plt.suptitle('Stationary marginal PDFs — CM (2006) 8D expansion', fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  → {save_path}")
    else:
        plt.show()


def plot_moments_vs_gamma(mg_results, save_path):
    """Plot <x²> and decay rate vs Gamma."""
    gammas = [r['Gamma'] for r in mg_results]
    var_A11 = [r['moments']['var_x'][0] for r in mg_results]
    var_A12 = [r['moments']['var_x'][1] for r in mg_results]
    decay   = [-r['lam_stat'].real for r in mg_results]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax1, ax2, ax3 = axes

    ax1.plot(gammas, var_A11, 'o-b', ms=7, lw=2, label='⟨A₀₀²⟩')
    ax1.plot(gammas, var_A12, 's-r', ms=7, lw=2, label='⟨A₀₁²⟩')
    ax1.axhline(D_MAT[0,0], color='b', ls=':', alpha=0.5, label='D₀₀ (OU pred)')
    ax1.axhline(D_MAT[1,1], color='r', ls=':', alpha=0.5, label='D₁₁ (OU pred)')
    ax1.set(xlabel='Γ', ylabel='Variance', title='Component variances')
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    ax2.plot(gammas, decay, 'D-g', ms=7, lw=2)
    ax2.set(xlabel='Γ', ylabel='First decay rate (−Re λ₁)',
            title='Slowest relaxation rate')
    ax2.grid(True, alpha=0.3)

    ax3.scatter([r['lam_stat'].real for r in mg_results],
                [r['Gamma'] for r in mg_results],
                c='steelblue', s=60)
    ax3.axvline(0, color='r', ls='--', lw=1)
    ax3.set(xlabel='Re(λ_stationary)', ylabel='Γ',
            title='Stationary eigenvalue (→0 = converged)')
    ax3.grid(True, alpha=0.3)

    plt.suptitle('CM (2006) 8D FPE — Moments and decay rates', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {save_path}")


# ─────────────────────────────────────────────────────────────────
# 14.  MAIN
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    OUT = 'fpe_8d_outputs'
    os.makedirs(OUT, exist_ok=True)
    warnings.filterwarnings('ignore')

    print("=" * 68)
    print("Chevillard-Meneveau FPE — 8D Complete Set Expansion")
    print("Risken §6.6.5, principal-coordinate Hermite basis")
    print("=" * 68)

    # ── Geometry report ─────────────────────────────────────────
    print(f"\nDiffusion matrix eigenvalues Λ = {np.round(D_EIGVALS, 4)}")
    print(f"Hermite scales a_α = 1/√Λ_α  = {np.round(1/np.sqrt(D_EIGVALS), 4)}")

    # ── Step 0: OU verification ──────────────────────────────────
    print("\n[0] OU exact verification (n_max=2, n_quad=4)")
    err = verify_ou_limit(n_max=2, n_quad=4, verbose=True)

    # ── Step 1: Convergence in n_max ────────────────────────────
    print("\n[1] Convergence study  (Γ=0.1, n_quad=4)")
    conv = convergence_study(
        Gamma=0.1,
        n_max_list=[1, 2, 3],
        n_quad=4,
        verbose=True
    )
    plot_convergence(conv, f'{OUT}/convergence.png')

    # ── Step 2: Multi-Gamma study ────────────────────────────────
    print("\n[2] Multi-Γ study  (n_max=2, n_quad=4)")
    mg = multi_gamma_study(
        gammas=[0.20, 0.10, 0.08, 0.06],
        n_max=2,
        n_quad=4,
        verbose=True
    )
    plot_eigenvalue_spectrum(mg, f'{OUT}/spectra.png')
    plot_marginal_pdfs(mg, comp=0, save_path=f'{OUT}/marginal_A11.png')
    plot_moments_vs_gamma(mg, f'{OUT}/moments_vs_gamma.png')

    # ── Step 3: Higher resolution run ───────────────────────────
    print("\n[3] Higher resolution  (Γ=0.1, n_max=3, n_quad=4)")
    L, basis = build_fpe_matrix(Gamma=0.1, n_max=3, n_quad=4, verbose=True)
    vals, vecs = solve_eigenvalues(L)
    lam_stat, c_stat = get_stationary(vals, vecs)
    a_vec = 1.0 / np.sqrt(D_EIGVALS)
    moms  = compute_moments(c_stat, basis, a_vec)

    print(f"\n  Stationary eigenvalue:  {lam_stat.real:.4e}")
    print(f"  N_basis = {len(basis)}")
    print(f"\n  First 10 decay rates −Re(λ_k):")
    for k in range(min(10, len(vals))):
        print(f"    k={k}: {-vals[k].real:.4f} + {vals[k].imag:.4f}i")
    print(f"\n  Moments (original x-coordinates):")
    comp_names = ['A₀₀','A₀₁','A₀₂','A₁₀','A₁₁','A₁₂','A₂₀','A₂₁']
    for alpha in range(NDIM):
        print(f"    <{comp_names[alpha]}> = {moms['mean_x'][alpha]:+.4f}   "
              f"<{comp_names[alpha]}²> = {moms['var_x'][alpha]:.4f}   "
              f"(OU pred: {D_MAT[alpha,alpha]:.4f})")

    print("\n" + "=" * 68)
    print("Outputs:")
    print(f"  {OUT}/convergence.png        — λ_stat vs n_max")
    print(f"  {OUT}/spectra.png            — eigenvalue spectra at 4 Γ")
    print(f"  {OUT}/marginal_A11.png       — marginal PDF of A₀₀")
    print(f"  {OUT}/moments_vs_gamma.png   — variances & decay rates")
    print("=" * 68)