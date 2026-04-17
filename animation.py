import numpy as np
import os
from orientation_2 import integrate_orientation

# --- your previously defined functions ---
# integrate_orientation(...)
# plot_rotating_ellipsoid_with_stress(...)
# generate_ellipsoid_points(...)
# compute_frobenius_norms(...)

# -------------------------
# 1. Load grad_u.csv
# -------------------------
data = np.loadtxt("grad_u.csv", delimiter=',', skiprows=1)
steps = data[:, 0]                  # time column
vec_A_lab = data[:, 1:10]           # 9 columns A11..A33
T = len(steps)

# reshape to (T,3,3) in row-major order
vec_A_lab = vec_A_lab.reshape(T, 3, 3)

# -------------------------
# 2. Define ellipsoid axes
# -------------------------
a = np.array([1.0, 0.5, 0.3])  # example axes

# -------------------------
# 3. Generate surface points for stress map
# -------------------------
def generate_ellipsoid_points(a, N_pts=500):
    # Use spherical sampling
    phi = np.arccos(2*np.random.rand(N_pts)-1)
    theta = 2*np.pi*np.random.rand(N_pts)
    x = a[0]*np.sin(phi)*np.cos(theta)
    y = a[1]*np.sin(phi)*np.sin(theta)
    z = a[2]*np.cos(phi)
    return np.stack([x,y,z], axis=1)

N_pts = 500
surface_points = generate_ellipsoid_points(a, N_pts=N_pts)

# -------------------------
# 4. Prepare transfer matrix M_grid for all points
#    (this comes from your previous stress computation)
# -------------------------
# For demonstration, assume M_grid is precomputed:
# M_grid.shape = (N_pts, 9, 9)
# Replace with your actual calculation
M_grid = np.random.rand(N_pts, 9, 9)  # placeholder

# -------------------------
# 5. Precompute Frobenius norms
# -------------------------
def compute_frobenius_norms(M_grid, vec_A_body, steps, print_every=1000, batch_size=None):
    T = len(steps)
    N_pts = M_grid.shape[0]
    frob_all = np.zeros((T, N_pts))

    # flatten 3x3 -> 9
    vec_A_body_flat = vec_A_body.reshape(T, 9)

    if batch_size is None:
        batch_size = T  # process all at once if possible

    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        vs_all = np.einsum('pij,tj->pti', M_grid, vec_A_body_flat[start:end])  # (N_pts, batch, 9)
        frob_all[start:end] = np.linalg.norm(vs_all, axis=2).T
        print(f"Processed {end}/{T} timesteps ({100*end/T:.1f}%)")

    return frob_all
# If vec_A_lab is in lab frame, convert to body frame if needed
# For simplicity, assume small initial rotation (identity)
vec_A_body = vec_A_lab.copy()

print("Precomputing stress norms...")
frob_all = compute_frobenius_norms(M_grid, vec_A_body, steps, print_every=1000) #vec_A_lab instead?
print("Finished stress norms.")

# -------------------------
# 6. Integrate orientation
# -------------------------
print("Integrating orientation...")
R_history, _, _, _, _ = integrate_orientation(a, vec_A_body, steps)
print("Finished integration.")

# -------------------------
# 7. Animate
# -------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from matplotlib import cm
import sys

def _ellipsoid_wireframe(a, R, n_theta=40, n_phi=20):
    """Return list of (N,3) points for ellipsoid wireframe (latitude + longitude)."""
    lines = []

    # Latitude rings (constant phi)
    for phi in np.linspace(0, np.pi, n_phi):
        theta = np.linspace(0, 2 * np.pi, n_theta)
        pts = np.stack([
            a[0] * np.sin(phi) * np.cos(theta),
            a[1] * np.sin(phi) * np.sin(theta),
            a[2] * np.cos(phi) * np.ones_like(theta),
        ], axis=1)
        pts = pts @ R.T
        lines.append(pts)

    # Longitude lines (constant theta)
    for theta in np.linspace(0, 2 * np.pi, n_theta, endpoint=False):
        phi = np.linspace(0, np.pi, n_phi)
        pts = np.stack([
            a[0] * np.sin(phi) * np.cos(theta),
            a[1] * np.sin(phi) * np.sin(theta),
            a[2] * np.cos(phi),
        ], axis=1)
        pts = pts @ R.T
        lines.append(pts)

    return lines

def _project(pts, plane):
    """Project (N,3) points onto 2D plane."""
    if plane == 'xy': return pts[:,0], pts[:,1]
    if plane == 'yz': return pts[:,1], pts[:,2]
    if plane == 'xz': return pts[:,0], pts[:,2]

def plot_rotating_ellipsoid_with_stress(
    a,
    R_history,
    steps,
    surface_points,
    stress_norms,
    save_every=1,
    output_path="ellipsoid_stress.mp4",
    fps=60,
    dpi=120
):
    """
    Animate ellipsoid rotation with stress heatmap on surface points.

    Parameters
    ----------
    a : array-like, (3,) - ellipsoid semi-axes
    R_history : (T,3,3) - rotation matrices
    steps : (T,) - time
    surface_points : (N_pts,3) - points on ellipsoid body frame
    stress_norms : (T,N_pts) - Frobenius norm at each surface point
    save_every : int - subsample timesteps for animation
    output_path : str - filename for mp4/gif
    """
    a = np.array(a)
    sp = np.array(surface_points)
    T = len(steps)
    N_pts = sp.shape[0]
    frame_indices = list(range(0, T, save_every))
    n_frames = len(frame_indices)

    planes = ['xy', 'yz', 'xz']
    plane_labels = [('x','y'),('y','z'),('x','z')]
    axis_colours = ['tab:red','tab:green','tab:blue']

    lim = max(a)*1.35

    fig, axes = plt.subplots(1,3,figsize=(13,4.5))
    fig.patch.set_facecolor('#0e0e0e')
    for ax in axes:
        ax.set_facecolor('#1a1a1a')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.axhline(0,color='#333',lw=0.5)
        ax.axvline(0,color='#333',lw=0.5)
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    title_text = fig.suptitle("", color='white', fontsize=11)

    # Wireframe artists
    dummy_lines = _ellipsoid_wireframe(a, np.eye(3))
    wire_artists = [[] for _ in range(3)]
    for j, ax in enumerate(axes):
        for line_pts in dummy_lines:
            px, py = _project(line_pts, planes[j])
            ln, = ax.plot(px, py, color='#5599ff', lw=0.4, alpha=0.55)
            wire_artists[j].append(ln)

    # Surface scatter points
    # Surface scatter points — initialize WITH data so vmin/vmax and array size are set
    scatters = []
    sp_init = sp @ np.eye(3)  # identity rotation at t=0
    for j, ax in enumerate(axes):
        px0, py0 = _project(sp_init, planes[j])
        sc = ax.scatter(
            px0, py0,
            c=stress_norms[0],       # ← initialize with actual data, not []
            s=20,
            cmap='inferno',
            vmin=np.percentile(stress_norms, 2),   # ← robust colormap limits
            vmax=np.percentile(stress_norms, 98)
        )
        scatters.append(sc)

    # Function to update each frame
    def update(frame_num):
        t = frame_indices[frame_num]
        R = R_history[t]
        wf_lines = _ellipsoid_wireframe(a, R)

        # Wireframe update
        for j in range(3):
            for ln, line_pts in zip(wire_artists[j], wf_lines):
                px, py = _project(line_pts, planes[j])
                ln.set_data(px, py)

        # Surface points + heatmap
        sp_rot = sp @ R.T
        for j, sc in enumerate(scatters):
            px, py = _project(sp_rot, planes[j])
            sc.set_offsets(np.c_[px, py])
            sc.set_array(stress_norms[t])

        angle_deg = np.degrees(np.arccos(np.clip((np.trace(R)-1)/2, -1,1)))
        title_text.set_text(f"t={steps[t]:.3f}   rotation ≈ {angle_deg:.1f}°")

        # Print progress
        percent = 100 * (frame_num+1)/n_frames
        sys.stdout.write(f"\rRendering frame {frame_num+1}/{n_frames} ({percent:.1f}%)")
        sys.stdout.flush()

        artists = sum(wire_artists, []) + scatters + [title_text]
        return artists

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000//fps, blit=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    ext = os.path.splitext(output_path)[1].lower()
    if ext == '.gif':
        writer = animation.PillowWriter(fps=fps)
    else:
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800, extra_args=['-vcodec','libx264','-pix_fmt','yuv420p'])

    print(f"Saving animation ({n_frames} frames) to '{output_path}'...")
    ani.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"Done. Saved to '{output_path}'")


plot_rotating_ellipsoid_with_stress(
    a=a,
    R_history=R_history,
    steps=steps,
    surface_points=surface_points,
    stress_norms=frob_all,
    save_every=50,
    output_path="ellipsoid_stress.mp4"
)