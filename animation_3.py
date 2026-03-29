# this is the same as original but heat map isntead

import os
os.environ["JAX_PLATFORMS"] = "cpu"  # must be before JAX imports

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree

from orientation_2 import integrate_orientation
from jeffery4_2 import Ellipsoid

# ===========================================================================
# CONFIGURATION
# ===========================================================================

CSV_PATH   = "grad_u2.csv"
OUTPUT     = "ellipsoid_stress.mp4"

a  = np.array([2.0, 1.0, 1.0])
mu = 1.0

# Surface grid resolution (theta x phi parametric grid)
N_THETA = 40
N_PHI   = 40

# Ray-cast image resolution per panel (increase for higher quality)
IMG_W = 300
IMG_H = 300

COMPUTE_EVERY = 10   # stride over CSV timesteps
SAVE_EVERY    = 1   # animate every N-th computed timestep
FPS           = 30
DPI           = 120

CMAP         = "inferno"
EINSUM_CHUNK = 200   # surface points per chunk for frob_grid

# ===========================================================================
# SURFACE GRID
# ===========================================================================

def make_surface_grid(a, n_theta, n_phi):
    theta = np.linspace(0, np.pi,      n_theta)
    phi   = np.linspace(0, 2 * np.pi, n_phi)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    X = a[0] * np.sin(TH) * np.cos(PH)
    Y = a[1] * np.sin(TH) * np.sin(PH)
    Z = a[2] * np.cos(TH)
    xyz = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    return xyz   # (N, 3) in body frame

# ===========================================================================
# RAY-CAST HEATMAP RENDERER
# ===========================================================================

def raycast_panel(a, R, stress_vals, plane, img_w, img_h, lim, tree):
    """
    Orthographic ray-cast of the ellipsoid surface onto a 2D projection plane.

    For every pixel:
      1. Shoot a ray along the depth axis into the scene (body frame).
      2. Solve analytic ray-ellipsoid intersection.
      3. Find the nearest surface grid point via KD-tree.
      4. Colour by stress value; pixels with no hit stay NaN (transparent).

    Parameters
    ----------
    a           : (3,) semi-axes
    R           : (3,3) rotation matrix  body -> lab
    stress_vals : (N_pts,) stress at each body-frame surface point
    plane       : 'xy' | 'yz' | 'xz'
    img_w/h     : pixel dimensions
    lim         : half-extent of coordinate window
    tree        : cKDTree built once on body-frame surface points

    Returns
    -------
    img : (img_h, img_w) float array, NaN where no intersection
    """
    # Which lab-frame axes map to (horizontal, vertical, depth)
    ax_map = {'xy': (0, 1, 2), 'yz': (1, 2, 0), 'xz': (0, 2, 1)}
    i0, i1, i_perp = ax_map[plane]

    # Pixel centres in lab frame
    u = np.linspace(-lim, lim, img_w)
    v = np.linspace( -lim,lim, img_h)   
    UU, VV = np.meshgrid(u, v)           # (H, W)
    N_rays = img_w * img_h

    # Ray origins: laid out on the projection plane, offset far along depth axis
    orig_lab = np.zeros((N_rays, 3))
    orig_lab[:, i0]    = UU.ravel()
    orig_lab[:, i1]    = VV.ravel()
    orig_lab[:, i_perp] = 10.0 * max(a)

    # Ray direction: straight in along depth axis (same for all rays)
    dir_lab = np.zeros(3)
    dir_lab[i_perp] = -1.0

    # Transform into body frame  (x_body = R^T x_lab)
    Rt        = R.T                                         
    orig_body = orig_lab @ Rt.T   # (N_rays, 3)
    dir_body  = Rt @ dir_lab      # (3,)

    # Analytic ray-ellipsoid intersection:
    #   ||(orig + t*dir) / a||^2 = 1
    d = dir_body / a                            # (3,)
    o = orig_body / a[np.newaxis, :]            # (N_rays, 3)

    A_c = np.dot(d, d)                          # scalar
    B_c = 2.0 * (o @ d)                        # (N_rays,)
    C_c = np.sum(o ** 2, axis=1) - 1.0         # (N_rays,)

    disc = B_c ** 2 - 4.0 * A_c * C_c
    hit  = disc >= 0.0

    img_flat = np.full(N_rays, np.nan)

    if hit.any():
        t_hit     = (-B_c[hit] - np.sqrt(disc[hit])) / (2.0 * A_c)
        pts_hit   = orig_body[hit] + t_hit[:, np.newaxis] * dir_body[np.newaxis, :]
        _, idx    = tree.query(pts_hit, workers=-1)
        img_flat[hit] = stress_vals[idx]

    return img_flat.reshape(img_h, img_w)


# ===========================================================================
# WIREFRAME HELPER
# ===========================================================================

def ellipsoid_wireframe(a, R, n_theta=30, n_phi=15):
    lines = []
    for phi_v in np.linspace(0, np.pi, n_phi):
        theta_v = np.linspace(0, 2 * np.pi, n_theta)
        pts = np.stack([
            a[0] * np.sin(phi_v) * np.cos(theta_v),
            a[1] * np.sin(phi_v) * np.sin(theta_v),
            a[2] * np.cos(phi_v) * np.ones_like(theta_v),
        ], axis=1) @ R.T
        lines.append(pts)
    for theta_v in np.linspace(0, 2 * np.pi, n_theta, endpoint=False):
        phi_v = np.linspace(0, np.pi, n_phi)
        pts = np.stack([
            a[0] * np.sin(phi_v) * np.cos(theta_v),
            a[1] * np.sin(phi_v) * np.sin(theta_v),
            a[2] * np.cos(phi_v),
        ], axis=1) @ R.T
        lines.append(pts)
    return lines


def project(pts, plane):
    if plane == 'xy': return pts[:, 0], pts[:, 1]
    if plane == 'yz': return pts[:, 1], pts[:, 2]
    if plane == 'xz': return pts[:, 0], pts[:, 2]


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    # ── Load CSV ──────────────────────────────────────────────────────────────
    print(f"Loading {CSV_PATH} ...")
    data       = np.loadtxt(CSV_PATH, delimiter=',', skiprows=1)
    steps_full = data[:, 0]
    A_full     = data[:, 1:10].reshape(-1, 3, 3)
    T_full     = len(steps_full)
    print(f"  {T_full} timesteps in CSV")

    idx      = np.arange(0, T_full, COMPUTE_EVERY)
    steps    = steps_full[idx]
    A_series = A_full[idx]
    T        = len(steps)
    print(f"  Using {T} timesteps (COMPUTE_EVERY={COMPUTE_EVERY})")

    # ── Integrate orientation ─────────────────────────────────────────────────
    print("Integrating orientation ODE ...")
    R_history, *_ = integrate_orientation(a, A_series, steps)

    tmp        = np.einsum('tji,tjk->tik', R_history, A_series)
    A_body_ts  = np.einsum('tij,tjk->tik', tmp, R_history)
    vec_A_body = A_body_ts.reshape(T, 9)

    # ── Surface grid & transfer matrices (once) ───────────────────────────────
    print("Building surface grid ...")
    xyz_body = make_surface_grid(a, N_THETA, N_PHI)
    N_pts    = len(xyz_body)
    print(f"  {N_THETA}x{N_PHI} = {N_pts} surface points")

    print("Building transfer matrices (vectorised batch) ...")
    ell   = Ellipsoid(a, np.zeros((3, 3)), mu=mu)
    ell.use_surface_mode()
    M_all = ell.build_transfer_matrices_batch(xyz_body)   # (N_pts, 9, 9)

    # KD-tree built once on body-frame points — rotation applied to rays instead
    tree = cKDTree(xyz_body)

    # ── Precompute frob_grid (N_pts, T) ──────────────────────────────────────
    print("Computing ||sigma||_F for all points x all timesteps ...")
    frob_grid = np.empty((N_pts, T), dtype=np.float64)
    A_T       = vec_A_body.T
    for start in range(0, N_pts, EINSUM_CHUNK):
        end  = min(start + EINSUM_CHUNK, N_pts)
        vs   = M_all[start:end] @ A_T
        frob_grid[start:end] = np.linalg.norm(vs, axis=1)
    print(f"  Done — shape {frob_grid.shape}, {frob_grid.nbytes/1e6:.1f} MB")

    vmin = float(np.nanpercentile(frob_grid, 2))
    vmax = float(np.nanpercentile(frob_grid, 98))
    norm = Normalize(vmin=vmin, vmax=vmax)
    print(f"  Stress range [{vmin:.3g}, {vmax:.3g}]")

    lim = max(a) * 1.3

    # ── Figure setup ──────────────────────────────────────────────────────────
    planes      = ['xy', 'yz', 'xz']
    plane_xlbls = ['x',  'y',  'x']
    plane_ylbls = ['y',  'z',  'z']

    fig, axes = plt.subplots(1, 3, figsize=(13, 6))
    fig.patch.set_facecolor('#0e0e0e')

    img_artists  = []
    wire_artists = [[] for _ in range(3)]

    for j, ax in enumerate(axes):
        ax.set_facecolor('#0e0e0e')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_xlabel(plane_xlbls[j], color='white', fontsize=9)
        ax.set_ylabel(plane_ylbls[j], color='white', fontsize=9)
        ax.tick_params(colors='white', labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')
        ax.axhline(0, color='#333', lw=0.5, zorder=0)
        ax.axvline(0, color='#333', lw=0.5, zorder=0)

        # Heatmap placeholder — updated each frame
        blank = np.full((IMG_H, IMG_W), np.nan)
        im = ax.imshow(
            blank,
            extent=[-lim, lim, -lim, lim],
            origin='lower',
            cmap=CMAP,
            norm=norm,
            interpolation='bilinear',
            aspect='equal',
            zorder=1,
        )
        img_artists.append(im)

        # Wireframe overlay drawn on top
        for line_pts in ellipsoid_wireframe(a, np.eye(3)):
            px, py = project(line_pts, planes[j])
            ln, = ax.plot(px, py, color='white', lw=0.35, alpha=0.25, zorder=2)
            wire_artists[j].append(ln)

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.03]) 
    cbar = fig.colorbar(sm, cax=cbar_ax, shrink=0.8, pad=0.04, orientation = 'horizontal', location = 'bottom',
                        label=r"$\|\sigma\|_F$")
    cbar.ax.xaxis.label.set_color('white')
    cbar.ax.tick_params(colors='white')

    title = fig.suptitle("", color='white', fontsize=11)
    fig.tight_layout()

    frame_indices = list(range(0, T, SAVE_EVERY))
    n_frames      = len(frame_indices)

    # ── Animation update ──────────────────────────────────────────────────────
    def update(frame_num):
        ti          = frame_indices[frame_num]
        R           = R_history[ti]
        stress_vals = frob_grid[:, ti]   # free slice — all heavy work done

        # Ray-cast each projection panel
        for j, plane in enumerate(planes):
            img = raycast_panel(a, R, stress_vals, plane,
                                IMG_W, IMG_H, lim, tree)
            img_artists[j].set_data(img)

        # Update wireframe
        wf = ellipsoid_wireframe(a, R)
        for j in range(3):
            for ln, line_pts in zip(wire_artists[j], wf):
                px, py = project(line_pts, planes[j])
                ln.set_data(px, py)

        angle_deg = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        title.set_text(
            f"t = {steps[ti]:.4g}   rotation ≈ {angle_deg:.1f}°   "
            f"frame {frame_num+1}/{n_frames}"
        )

        sys.stdout.write(
            f"\r  frame {frame_num+1}/{n_frames}  "
            f"({100*(frame_num+1)/n_frames:.1f}%)"
        )
        sys.stdout.flush()

        return img_artists + sum(wire_artists, []) + [title]

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames,
        interval=1000 // FPS, blit=True,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    ext = os.path.splitext(OUTPUT)[1].lower()
    if ext == '.gif':
        writer = animation.PillowWriter(fps=FPS)
    else:
        writer = animation.FFMpegWriter(
            fps=FPS, bitrate=2400,
            extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'],
        )

    print(f"\nSaving {n_frames} frames -> {OUTPUT}")
    ani.save(OUTPUT, writer=writer, dpi=DPI)
    plt.close(fig)
    print(f"\nDone -> {OUTPUT}")


if __name__ == "__main__":
    main()