"""
generate_grad_u_LW.py  (optimised)
====================================
Leppin & Wilczek (2020) stochastic velocity-gradient model.

Optimisations vs original:
  1. float32 ensemble  — halves memory bandwidth for all (N,3,3) ops
  2. S, W, eps computed once per step and reused in both drift and
     coeff update — eliminates redundant computation
  3. Noise drawn in one randn call per step rather than multiple
  4. np.random.default_rng used explicitly for speed
  5. project_traceless uses vectorised diagonal subtraction, no loop

Physics and all CONFIG values unchanged.
"""

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# CONFIG  (unchanged from original)
# ─────────────────────────────────────────────────────────────

ALPHA      = -0.6
GAMMA      = -1.1
SIGMA      = 0.08
EPS_COEF   = -1e-8

N_ENS      = 1000
DT         = 0.0002
T_TRANS    = 100.0
T_SIM      = 200.0
SAVE_EVERY = 5

OUT_CSV    = "grad_u_LW.csv"
COEFF_UPDATE_EVERY = 50

# float32 for ensemble — sufficient precision, ~2x faster memory ops
DTYPE = np.float32

# ─────────────────────────────────────────────────────────────
# TENSOR HELPERS  — (N,3,3) batches
# ─────────────────────────────────────────────────────────────

_I3 = np.eye(3, dtype=DTYPE)

def sym(M):
    return 0.5 * (M + M.transpose(0, 2, 1))

def asym(M):
    return 0.5 * (M - M.transpose(0, 2, 1))

def tr_b(M):
    return np.einsum('nii->n', M)

def mm(A, B):
    return np.einsum('nij,njk->nik', A, B)

def traceless(M):
    t = tr_b(M)
    return M - (t[:, None, None] / 3.0) * _I3

def emean(x):
    return float(np.mean(x))

# ─────────────────────────────────────────────────────────────
# NONLINEAR DAMPING
# ─────────────────────────────────────────────────────────────

def eps_per_particle(S, W):
    TrW2 = tr_b(mm(W, W))
    TrS2 = tr_b(mm(S, S))
    return EPS_COEF * ((TrW2 + 0.5)**4 + (TrS2 - 0.5)**4)

# ─────────────────────────────────────────────────────────────
# ADAPTIVE COEFFICIENTS  (SM Eqs. S5-S7)
# Takes precomputed S, W, eps — no redundant recomputation
# ─────────────────────────────────────────────────────────────

def compute_adaptive_coeffs(A, S, W, eps):
    W2   = mm(W, W)
    A2   = mm(A, A)
    A3   = mm(A2, A)
    S2   = mm(S, S)
    SW2  = mm(S, W2)

    mTrSW2 = emean(tr_b(SW2))

    eps_TrW2 = emean(eps * tr_b(W2))
    eps_TrA2 = emean(eps * tr_b(A2))
    eps_TrA3 = emean(eps * tr_b(A3))

    A4       = mm(A2, A2)
    TrA2A2tl = tr_b(A4) - tr_b(A2)**2 / 3.0
    TrA2S2tl = tr_b(mm(A2, S2)) - tr_b(A2)*tr_b(S2)/3.0
    TrA2S    = tr_b(mm(A2, S))
    TrA2W2tl = tr_b(mm(A2, W2)) - tr_b(A2)*tr_b(W2)/3.0

    mTrA2A2tl = emean(TrA2A2tl)
    mTrA2S2tl = emean(TrA2S2tl)
    mTrA2S    = emean(TrA2S)
    mTrA2W2tl = emean(TrA2W2tl)

    # xi (S5)
    xi = 2.0*eps_TrW2 - (15.0/2.0)*SIGMA**2 - 4.0*mTrSW2

    # beta (S7)
    num_b = (mTrA2A2tl
             + ALPHA*(mTrA2S2tl + 6.0*mTrSW2*mTrA2S)
             + 2.0*eps_TrA2*mTrA2S
             - eps_TrA3)
    den_b = 2.0*mTrSW2*mTrA2S - mTrA2W2tl
    beta  = float(np.clip(num_b / den_b, -2.0, 2.0)) \
            if abs(den_b) > 1e-14 else -0.41

    # delta (S6)
    delta = 2.0*mTrSW2*(3.0*ALPHA - beta) + 2.0*eps_TrA2

    return beta, delta, xi

# ─────────────────────────────────────────────────────────────
# NOISE  (SM Eq. S8) — single randn call
# ─────────────────────────────────────────────────────────────

def sample_noise(N, dt, rng):
    raw = rng.standard_normal((N, 3, 3)).astype(DTYPE) \
          * DTYPE(SIGMA * np.sqrt(dt))
    return traceless(sym(raw)) + asym(raw)

# ─────────────────────────────────────────────────────────────
# DRIFT — takes precomputed S, W, eps
# ─────────────────────────────────────────────────────────────

def drift(A, S, W, eps, beta, delta, xi):
    A2tl = traceless(mm(A, A))
    S2tl = traceless(mm(S, S))
    W2tl = traceless(mm(W, W))

    H_e  = (ALPHA * S2tl
            + beta  * W2tl
            + GAMMA * (mm(S, W) - mm(W, S))
            + delta * S)

    damp = (xi + eps[:, None, None]) * A
    return -A2tl - H_e + damp

# ─────────────────────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────────────────────

def initialise_ensemble(N, rng):
    A = rng.standard_normal((N, 3, 3)).astype(DTYPE)
    t = tr_b(A)
    A[:, 0, 0] -= t / 3.0
    A[:, 1, 1] -= t / 3.0
    A[:, 2, 2] -= t / 3.0
    S    = sym(A)
    TrS2 = float(np.mean(tr_b(mm(S, S))))
    A   *= DTYPE(np.sqrt(0.5 / max(TrS2, 1e-12)))
    return A

def project_traceless_inplace(A):
    t = (A[:, 0, 0] + A[:, 1, 1] + A[:, 2, 2]) / 3.0
    A[:, 0, 0] -= t
    A[:, 1, 1] -= t
    A[:, 2, 2] -= t

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def run():
    rng = np.random.default_rng(42)

    n_trans = int(T_TRANS / DT)
    n_sim   = int(T_SIM   / DT)
    n_total = n_trans + n_sim
    n_save  = n_sim // SAVE_EVERY

    print("=" * 60)
    print("Leppin-Wilczek (2020) velocity gradient model")
    print(f"  alpha={ALPHA}, gamma={GAMMA}, sigma={SIGMA}")
    print(f"  N_ensemble={N_ENS},  dt={DT}")
    print(f"  Transient: {T_TRANS} tau_eta  ({n_trans:,} steps)")
    print(f"  Production: {T_SIM} tau_eta  ({n_sim:,} steps)")
    print(f"  Output: {n_save:,} rows  ->  {OUT_CSV}")
    print("=" * 60)

    A = initialise_ensemble(N_ENS, rng)
    beta, delta, xi = -0.41, -0.089, -0.17

    saved_A    = np.zeros((n_save, 3, 3), dtype=np.float64)
    saved_time = np.zeros(n_save)
    save_idx   = 0
    t          = 0.0

    # Compute S, W, eps once before loop — reused each step
    S   = sym(A)
    W   = asym(A)
    eps = eps_per_particle(S, W)

    for step in range(n_total):

        # Update coefficients using already-computed S, W, eps
        if step % COEFF_UPDATE_EVERY == 0:
            beta, delta, xi = compute_adaptive_coeffs(A, S, W, eps)

        # Step — drift uses same S, W, eps (no recomputation)
        dA = drift(A, S, W, eps, beta, delta, xi) * DT \
             + sample_noise(N_ENS, DT, rng)
        A  = A + dA
        project_traceless_inplace(A)
        t += DT

        # Update S, W, eps for next step (one computation per step)
        S   = sym(A)
        W   = asym(A)
        eps = eps_per_particle(S, W)

        if step == n_trans - 1:
            print("Transient complete. Starting production run...")

        if step >= n_trans and (step - n_trans) % SAVE_EVERY == 0:
            saved_A[save_idx]    = A[0].astype(np.float64)
            saved_time[save_idx] = t
            save_idx += 1

        if step % 500000 == 0:
            TrS2 = float(np.mean(tr_b(mm(S, S))))
            TrA3 = float(np.mean(tr_b(mm(mm(A, A), A))))
            tag  = "TRANS" if step < n_trans else "PROD "
            print(f"  [{tag}] step={step:>8,}  t={t:7.1f}  "
                  f"<Tr(S2)>={TrS2:.4f}  <Tr(A3)>={TrA3:+.4f}  "
                  f"beta={beta:.3f} delta={delta:.3f} xi={xi:.3f}")

    print(f"\nProduction done. Saved {save_idx} rows.")

    cols = ['time','A11','A12','A13','A21','A22','A23','A31','A32','A33']
    data = np.column_stack([
        saved_time[:save_idx],
        saved_A[:save_idx].reshape(save_idx, 9)
    ])
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved '{OUT_CSV}'  ({len(df)} rows,  "
          f"t = {df['time'].iloc[0]:.3f} -> {df['time'].iloc[-1]:.3f})")

    # Final diagnostics on saved float64 trajectory
    A_all = saved_A[:save_idx]
    S_all = 0.5*(A_all + A_all.transpose(0,2,1))
    W_all = 0.5*(A_all - A_all.transpose(0,2,1))
    TrS2  = np.einsum('tij,tij->t', S_all, S_all)
    TrW2  = np.einsum('tij,tij->t', W_all, W_all)
    A2    = np.einsum('tij,tjk->tik', A_all, A_all)
    TrA2  = np.einsum('tii->t', A2)
    TrA3  = np.einsum('tij,tji->t', A2, A_all)
    print("\nFinal diagnostics (single trajectory):")
    print(f"  <Tr(S2)> = {np.mean(TrS2):.4f}  (target  0.5000)")
    print(f"  <Tr(W2)> = {-np.mean(TrW2):.4f}  (target -0.5000)")
    print(f"  <Tr(A2)> = {np.mean(TrA2):.4f}  (target  0.0000)  [Betchov 1]")
    print(f"  <Tr(A3)> = {np.mean(TrA3):.4f}  (target  0.0000)  [Betchov 2]")


if __name__ == '__main__':
    run()