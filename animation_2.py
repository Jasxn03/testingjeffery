import os
os.environ["JAX_PLATFORMS"] = "cpu"  # must be set before JAX is imported

"""
animate_part_c.py
=================
Animates two Part-C plots driven by the actual grad_u.csv time series.

  part_c_3d_mean_anim.mp4      — 3D ellipsoid coloured by ||sigma||_F at each t
  part_c_2d_summaries_anim.mp4 — 2x2 (theta,phi) heatmaps at each t

Strategy
--------
  1. Build M_all (N_pts, 9, 9) ONCE via build_transfer_matrices_batch.
  2. Compute the full frob_grid (N_pts, T) ONCE:
         vec_sigma[pt, :, t] = M_all[pt] @ vec_A_body[t]
         frob_grid[pt, t]    = ||vec_sigma[pt, :, t]||
     Chunked einsum — pure numpy, no MC, no GMM.
  3. Each animation frame is a free slice frob_grid[:, ti].

Speed controls (set in CONFIGURATION or via CLI flags)
------------------------------------------------------
  --compute-every N   stride when slicing the time axis of the CSV
                      (e.g. 5 = use every 5th timestep; default 1)
  --save-every    N   further stride when writing frames to the video
                      (e.g. 10 = write every 10th computed frame; default 1)
  --fps           N   output video frame rate

Example: --compute-every 2 --save-every 5 --fps 15

Usage
-----
  python animate_part_c.py
  python animate_part_c.py --csv my_data.csv --compute-every 2 --save-every 5 --fps 12
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")   # prevent JAX from probing for CUDA

import argparse
import time as _time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FFMpegWriter
import pandas as pd

from jeffery4_2 import Ellipsoid
from orientation_2 import integrate_orientation

# ===========================================================================
# CONFIGURATION  <- edit here, or override via CLI flags
# ===========================================================================

CSV_PATH      = "grad_u.csv"
OUT_DIR       = "."

COMPUTE_EVERY = 100    # stride over CSV timesteps
SAVE_EVERY    = 1     # stride over computed frames when writing video
FPS           = 5    # output video frames per second

a  = np.array([2.0, 1.0, 1.0])   # ellipsoid semi-axes
mu = 1.0                           # viscosity

N_THETA = 80
N_PHI   = 80

EXCEEDANCE_THRESHOLD = 10.0   # for P(||sigma||_F > c) panel
N_CONTOURS           = 6
RADIAL_SCALE         = 0.25   # max radial bump as fraction of smallest semi-axis

CMAPS = {
    "mean":       "viridis",
    "std":        "plasma",
    "p90":        "inferno",
    "exceedance": "magma",
}
LABELS = {
    "mean":       r"$\|\sigma\|_F$",
    "std":        r"$\|\sigma\|_F$  (zero for single snapshot)",
    "p90":        r"$\|\sigma\|_F$  (= mean for single snapshot)",
    "exceedance": rf"$\|\sigma\|_F > {EXCEEDANCE_THRESHOLD}$  (0/1)",
}

EINSUM_CHUNK = 200   # surface points processed per chunk — tune to your RAM

# ===========================================================================
# HELPERS
# ===========================================================================

def ellipsoid_surface_grid(a, n_theta, n_phi):
    theta = np.linspace(0, np.pi,      n_theta)
    phi   = np.linspace(0, 2 * np.pi, n_phi)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    X = a[0] * np.sin(TH) * np.cos(PH)
    Y = a[1] * np.sin(TH) * np.sin(PH)
    Z = a[2] * np.cos(TH)
    xyz = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    return xyz, theta, phi, X, Y, Z


def normalise_01(F):
    lo, hi = np.nanpercentile(F, 2), np.nanpercentile(F, 98)
    if hi == lo:
        return np.zeros_like(F)
    return np.clip((F - lo) / (hi - lo), 0, 1)

# ===========================================================================
# MAIN
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",           default=CSV_PATH)
    p.add_argument("--fps",           type=int, default=FPS)
    p.add_argument("--compute-every", type=int, default=COMPUTE_EVERY,
                   help="Stride over CSV timesteps")
    p.add_argument("--save-every",    type=int, default=SAVE_EVERY,
                   help="Stride over computed frames when writing to video")
    p.add_argument("--out-dir",       default=OUT_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── 1. Load CSV ────────────────────────────────────────────────────────
    print(f"Loading {args.csv} ...")
    df     = pd.read_csv(args.csv)
    t_full = df["time"].values
    T_full = len(t_full)

    A_full = np.zeros((T_full, 3, 3))
    for i in range(3):
        for j in range(3):
            A_full[:, i, j] = df[f"A{i+1}{j+1}"].values
    print(f"  {T_full} timesteps in CSV")

    # Apply compute stride
    idx = np.arange(0, T_full, args.compute_every)
    t        = t_full[idx]
    A_series = A_full[idx]
    T        = len(t)
    print(f"  Using {T} timesteps  (--compute-every {args.compute_every})")

    # ── 2. Integrate orientation -> body-frame A ───────────────────────────
    print("Integrating orientation ODE ...")
    t0 = _time.perf_counter()
    R_history, *_ = integrate_orientation(a, A_series, t)
    print(f"  Done in {_time.perf_counter()-t0:.2f}s")

    tmp        = np.einsum('tji,tjk->tik', R_history, A_series)
    A_body_ts  = np.einsum('tij,tjk->tik', tmp, R_history)   # (T, 3, 3)
    vec_A_body = A_body_ts.reshape(T, 9)                      # (T, 9)

    # ── 3. Build surface grid & transfer matrices ONCE ────────────────────
    print("Building surface grid ...")
    xyz, theta, phi, X_ell, Y_ell, Z_ell = ellipsoid_surface_grid(a, N_THETA, N_PHI)
    N_pts = len(xyz)
    print(f"  {N_THETA}x{N_PHI} = {N_pts} surface points")

    print("Building transfer matrices (vectorised batch) ...")
    t0  = _time.perf_counter()
    ell = Ellipsoid(a, np.zeros((3, 3)), mu=mu)
    ell.use_surface_mode()
    M_all = ell.build_transfer_matrices_batch(xyz)   # (N_pts, 9, 9)
    print(f"  Done in {_time.perf_counter()-t0:.1f}s")

    # ── 4. Precompute frob_grid (N_pts, T) ONCE ───────────────────────────
    # vec_sigma = M_all @ vec_A_body.T  =>  (N_pts, 9, T)
    # frob      = ||vec_sigma||_2 over axis 1  =>  (N_pts, T)
    print("Computing ||sigma||_F for all points x all timesteps ...")
    t0        = _time.perf_counter()
    frob_grid = np.empty((N_pts, T), dtype=np.float64)
    A_T       = vec_A_body.T                              # (9, T)

    for start in range(0, N_pts, EINSUM_CHUNK):
        end  = min(start + EINSUM_CHUNK, N_pts)
        vs   = M_all[start:end] @ A_T                    # (chunk, 9, T)
        frob_grid[start:end] = np.linalg.norm(vs, axis=1)  # (chunk, T)

    print(f"  Done in {_time.perf_counter()-t0:.2f}s  "
          f"[shape {frob_grid.shape}, {frob_grid.nbytes/1e6:.1f} MB]")

    # Global colour limits fixed across all frames for consistent colour scale
    global_vmin = float(np.nanpercentile(frob_grid, 2))
    global_vmax = float(np.nanpercentile(frob_grid, 98))
    print(f"  Global ||sigma||_F range  [{global_vmin:.3g}, {global_vmax:.3g}]")

    # ── 5. Render animations ───────────────────────────────────────────────
    frame_indices = list(range(0, T, args.save_every))
    n_frames      = len(frame_indices)
    print(f"\nRendering {n_frames} frames  "
          f"(--save-every {args.save_every}, --fps {args.fps}) ...")

    out_3d = os.path.join(args.out_dir, "part_c_3d_mean_anim.mp4")
    out_2d = os.path.join(args.out_dir, "part_c_2d_summaries_anim.mp4")

    fig_3d = plt.figure(figsize=(9, 7))
    ax_3d  = fig_3d.add_subplot(111, projection='3d')

    fig_2d, axes_2d = plt.subplots(2, 2, figsize=(14, 10))

    # Pre-build outward unit normals (geometry is fixed)
    nx = X_ell / a[0]**2;  ny = Y_ell / a[1]**2;  nz = Z_ell / a[2]**2
    n_mag = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-15
    nx, ny, nz = nx / n_mag, ny / n_mag, nz / n_mag

    norm_3d   = plt.Normalize(vmin=global_vmin, vmax=global_vmax)
    theta_deg = np.degrees(theta)
    phi_deg   = np.degrees(phi)

    writer_3d = FFMpegWriter(fps=args.fps, metadata=dict(title="part_c_3d_mean"))
    writer_2d = FFMpegWriter(fps=args.fps, metadata=dict(title="part_c_2d_summaries"))

    t_render = _time.perf_counter()

    with writer_3d.saving(fig_3d, out_3d, dpi=120), \
         writer_2d.saving(fig_2d, out_2d, dpi=120):

        for fi, ti in enumerate(frame_indices):
            frob_col = frob_grid[:, ti]          # (N_pts,) — essentially free
            t_val    = t[ti]
            F        = frob_col.reshape(N_THETA, N_PHI)

            # ── 3D ────────────────────────────────────────────────────────
            ax_3d.cla()

            offset   = normalise_01(F) * RADIAL_SCALE * a.min()
            Xp = X_ell + offset * nx
            Yp = Y_ell + offset * ny
            Zp = Z_ell + offset * nz

            fcolours = plt.colormaps[CMAPS["mean"]](norm_3d(F))
            ax_3d.plot_surface(Xp, Yp, Zp, facecolors=fcolours,
                               rstride=1, cstride=1,
                               linewidth=0, antialiased=True, shade=False)

            ax_3d.set_xlabel(r"$x_1$"); ax_3d.set_ylabel(r"$x_2$")
            ax_3d.set_zlabel(r"$x_3$")
            ax_3d.set_title(
                r"$\|\sigma\|_F$  on ellipsoid surface"
                f"\nt = {t_val:.4g}   [frame {fi+1}/{n_frames}]",
                fontsize=11,
            )
            if fi == 0:
                sm = cm.ScalarMappable(cmap=CMAPS["mean"], norm=norm_3d)
                sm.set_array([])
                fig_3d._cb = fig_3d.colorbar(
                    sm, ax=ax_3d, shrink=0.5, pad=0.1,
                    label=r"$\|\sigma\|_F$",
                )
            fig_3d.tight_layout()
            writer_3d.grab_frame()

            # ── 2D ────────────────────────────────────────────────────────
            panel_fields = {
                "mean":       F,
                "std":        np.zeros_like(F),
                "p90":        F,
                "exceedance": (frob_col > EXCEEDANCE_THRESHOLD
                               ).astype(float).reshape(N_THETA, N_PHI),
            }

            for ax, (key, Fp) in zip(axes_2d.ravel(), panel_fields.items()):
                ax.cla()
                ax.pcolormesh(phi_deg, theta_deg, Fp,
                              cmap=CMAPS[key], shading='auto')
                cs = ax.contour(phi_deg, theta_deg, Fp,
                                levels=N_CONTOURS,
                                colors='white', linewidths=0.7, alpha=0.7)
                ax.clabel(cs, inline=True, fontsize=6, fmt='%.2g')
                ax.set_xlabel(r"$\phi$ (deg)")
                ax.set_ylabel(r"$\theta$ (deg)")
                ax.set_title(LABELS[key], fontsize=10)

            fig_2d.suptitle(
                r"$\|\sigma\|_F$ over ellipsoid surface"
                f"   t = {t_val:.4g}   [frame {fi+1}/{n_frames}]",
                fontsize=12,
            )
            fig_2d.tight_layout()
            writer_2d.grab_frame()

            # ── Progress (print ~20 times total) ──────────────────────────
            if (fi + 1) % max(1, n_frames // 20) == 0 or fi == 0:
                elapsed = _time.perf_counter() - t_render
                eta     = elapsed / (fi + 1) * (n_frames - fi - 1)
                print(f"  frame {fi+1:5d}/{n_frames}  t={t_val:.4g}  "
                      f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

    plt.close(fig_3d)
    plt.close(fig_2d)

    total = _time.perf_counter() - t_render
    print(f"\nDone in {total:.1f}s")
    print(f"  {out_3d}")
    print(f"  {out_2d}")


if __name__ == "__main__":
    main()