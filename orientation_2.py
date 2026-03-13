import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


def integrate_orientation(a, A_timeseries, steps, R0=None):
    """
    Integrate Jeffery's ODE to track ellipsoid orientation over time.

    dR/dt = skew(omega) @ R

    where omega is computed from the body-frame velocity gradient.

    Parameters
    ----------
    a : array-like, shape (3,)
        Semi-axes of the ellipsoid.
    A_timeseries : np.ndarray, shape (T, 3, 3)
        Lab-frame velocity gradient at each timestep.
    steps : np.ndarray, shape (T,)
        Time values (used to compute dt).
    R0 : np.ndarray, shape (3, 3), optional
        Initial rotation matrix. Defaults to identity (axes aligned with lab).

    Returns
    -------
    R_history : np.ndarray, shape (T, 3, 3)
        Rotation matrix at each timestep.
    omega_history : np.ndarray, shape (T, 3)
        Body-frame angular velocity at each timestep.
    angle_history : np.ndarray, shape (T,)
        Cumulative rotation angle (radians) from initial orientation.
    axis_history : np.ndarray, shape (T, 3)
        Instantaneous rotation axis (unit vector) at each timestep.
    """
    a = np.array(a)
    T = len(steps)

    R = np.eye(3) if R0 is None else np.array(R0)

    R_history     = np.zeros((T, 3, 3))
    omega_history = np.zeros((T, 3))
    angle_history = np.zeros(T)
    axis_history  = np.zeros((T, 3))

    cumulative_angle = 0.0
    dtheta_history = np.zeros((T,3))

    for t in range(T):
        # Transform A into body frame
        A_body = R.T @ A_timeseries[t] @ R

        # Decompose body-frame A
        E = 0.5 * (A_body + A_body.T)   # strain in body frame
        W = 0.5 * (A_body - A_body.T)   # vorticity in body frame

        # Vorticity vector from anti-symmetric part
        Omega_vec = np.array([W[2, 1], W[0, 2], W[1, 0]])

        # Jeffery's omega: vorticity corrected by strain
        omega = np.zeros(3)
        for i in range(3):
            i1 = (i+1) % 3; i2 = (i+2) % 3
            a1sq = a[i1]**2; a2sq = a[i2]**2
            omega[i] = Omega_vec[i] - (a1sq - a2sq) / (a1sq + a2sq) * E[i2, i1]

        # Store omega and instantaneous axis
        omega_mag = np.linalg.norm(omega)
        omega_history[t] = omega
        axis_history[t]  = omega / omega_mag if omega_mag > 1e-12 else np.array([0, 0, 1])

        # Store current R before stepping
        R_history[t] = R

        # Integrate: R_{t+1} = expm(skew(omega) * dt) @ R
        dt = steps[t+1] - steps[t] if t < T-1 else steps[-1] - steps[-2]
        dtheta = omega * dt
        dtheta_history[t] = omega * dt
        cumulative_angle += omega_mag * dt
        angle_history[t] = cumulative_angle

        angle = np.linalg.norm(dtheta)
        if angle > 1e-12:
            axis = dtheta / angle
            # Rodrigues' rotation formula for the update
            K = np.array([[     0, -axis[2],  axis[1]],
                          [ axis[2],      0, -axis[0]],
                          [-axis[1],  axis[0],      0]])
            dR = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
            R = dR @ R
            # Re-orthogonalise R periodically to prevent drift
            U, _, Vt = np.linalg.svd(R)
            R = U @ Vt

    return R_history, omega_history, angle_history, axis_history, dtheta_history


def plot_rotation_history(steps, rotation_history, output_dir="."):
    """
    Plot cumulative rotation angle and instantaneous rotation axis over time.

    Parameters
    ----------
    steps : np.ndarray, shape (T,)
        Time values.
    rotation_history : dict
        Output from integrate_orientation, with keys:
        'omega', 'angle', 'axis', 'R'
    output_dir : str
        Directory to save figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    omega = rotation_history['omega']   # (T, 3)
    angle = rotation_history['angle']   # (T,)
    axis  = rotation_history['axis']    # (T, 3)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    # --- Cumulative rotation angle ---
    ax = axes[0]
    ax.plot(steps, np.degrees(angle), color='black')
    ax.set_ylabel("Cumulative rotation (degrees)")
    ax.set_xlabel("Time")
    ax.set_title("Total rotation angle over time")
    ax.grid(True, alpha=0.3)

    # --- Instantaneous angular velocity components ---
    ax = axes[1]
    for k, label in enumerate(['$\\omega_1$ (body)', '$\\omega_2$ (body)', '$\\omega_3$ (body)']):
        ax.plot(steps, omega[:, k], label=label)
    ax.set_ylabel("Angular velocity")
    ax.set_xlabel("Time")
    ax.set_title("Instantaneous angular velocity (body frame)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Instantaneous rotation axis components ---
    ax = axes[2]
    for k, label in enumerate(['axis $e_1$', 'axis $e_2$', 'axis $e_3$']):
        ax.plot(steps, axis[:, k], label=label)
    ax.set_ylabel("Axis component")
    ax.set_xlabel("Time")
    ax.set_title("Instantaneous rotation axis direction")
    ax.set_ylim(-1.1, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(output_dir, "rotation_history.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


def plot_cumulative_vs_net_rotation(steps, rotation_history, output_dir="."):
    """
    Plot cumulative (path length) rotation vs net rotation angle over time.

    Cumulative angle: sum of |omega| * dt at each step — total path traversed
    in rotation space, always increasing.

    Net angle: angle between current orientation and initial orientation,
    computed from trace(R). This is what you actually see in the animation.

    Parameters
    ----------
    steps : np.ndarray, shape (T,)
        Time values.
    rotation_history : dict
        Output from integrate_orientation, with keys 'R' and 'angle'.
    output_dir : str
        Directory to save figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    R_history        = rotation_history['R']                           # (T, 3, 3)
    cumulative_angle = np.degrees(rotation_history['angle'])           # (T,)

    # Compute net angle from R at each timestep
    traces    = (np.trace(R_history, axis1=1, axis2=2) - 1) / 2       # (T,)
    net_angle = np.degrees(np.arccos(np.clip(traces, -1, 1)))          # (T,)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, cumulative_angle, label='Cumulative (path length)', color='tab:blue')
    ax.plot(steps, net_angle,        label='Net (from initial orientation)', color='tab:orange')
    ax.set_xlabel("Time")
    ax.set_ylabel("Angle (degrees)")
    ax.set_title("Cumulative vs net rotation angle")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(output_dir, "cumulative_vs_net_rotation.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


def _ellipsoid_wireframe(a, R, n_theta=40, n_phi=20):
    """
    Generate rotated ellipsoid wireframe lines.

    Returns a list of (x, y, z) arrays, each a single line on the surface.
    Longitude lines + latitude lines, rotated by R.
    """
    lines = []

    # Latitude rings (constant phi)
    for phi in np.linspace(0, np.pi, n_phi):
        theta = np.linspace(0, 2 * np.pi, n_theta)
        pts = np.stack([
            a[0] * np.sin(phi) * np.cos(theta),
            a[1] * np.sin(phi) * np.sin(theta),
            a[2] * np.cos(phi) * np.ones_like(theta),
        ], axis=1)           # (N, 3)
        pts = pts @ R.T      # rotate: (N, 3) @ (3, 3)
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
    """Return 2D projection of (N,3) pts onto plane 'xy', 'yz', or 'xz'."""
    if plane == 'xy':
        return pts[:, 0], pts[:, 1]
    elif plane == 'yz':
        return pts[:, 1], pts[:, 2]
    elif plane == 'xz':
        return pts[:, 0], pts[:, 2]


def plot_rotating_ellipsoid_animation(
    a,
    R_history,
    steps,
    surface_points=None,
    output_path="ellipsoid_rotation.mp4",
    fps=15,
    dpi=120,
    save_every=1,
):
    """
    Animate the rotating ellipsoid as three 2D projections (xy, yz, xz).

    Each frame shows:
      - Wireframe outline of the ellipsoid rotated by R_history[t]
      - Body axes e1, e2, e3 as coloured arrows
      - Surface points (if provided) rotated into lab frame

    Parameters
    ----------
    a : array-like, shape (3,)
        Semi-axes of the ellipsoid.
    R_history : np.ndarray, shape (T, 3, 3)
        Rotation matrix at each timestep (from integrate_orientation).
    steps : np.ndarray, shape (T,)
        Time values (used for title/label).
    surface_points : list of array-like, optional
        Points on ellipsoid surface (body frame); will be rotated each frame.
    output_path : str
        Path to save the MP4 (or GIF if path ends in .gif).
    fps : int
        Frames per second.
    dpi : int
        Resolution.
    save_every : int
        Use every nth timestep as a frame (reduces file size / speeds up).
    """
    a = np.array(a)
    T = len(steps)

    # Subsample timesteps
    frame_indices = list(range(0, T, save_every))
    n_frames = len(frame_indices)

    planes       = ['xy', 'yz', 'xz']
    plane_labels = [('x', 'y'), ('y', 'z'), ('x', 'z')]
    axis_colours = ['tab:red', 'tab:green', 'tab:blue']
    axis_labels  = ['$e_1$', '$e_2$', '$e_3$']

    lim = max(a) * 1.35

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.patch.set_facecolor('#0e0e0e')
    for ax in axes:
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')

    for ax, (xl, yl), plane in zip(axes, plane_labels, planes):
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_aspect('equal')
        ax.axhline(0, color='#333333', lw=0.5)
        ax.axvline(0, color='#333333', lw=0.5)

    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    title_text = fig.suptitle("", color='white', fontsize=11)

    wire_artists  = [[] for _ in range(3)]
    arrow_artists = [[] for _ in range(3)]
    pt_artists    = [None] * 3

    dummy_lines = _ellipsoid_wireframe(a, np.eye(3))
    for j, ax in enumerate(axes):
        for line_pts in dummy_lines:
            px, py = _project(line_pts, planes[j])
            ln, = ax.plot(px, py, color='#5599ff', lw=0.4, alpha=0.55)
            wire_artists[j].append(ln)

    for j, ax in enumerate(axes):
        for k in range(3):
            ln, = ax.plot([], [], color=axis_colours[k], lw=2.0,
                          label=axis_labels[k])
            arrow_artists[j].append(ln)
        if j == 0:
            ax.legend(loc='upper right', fontsize=7,
                      facecolor='#222222', labelcolor='white',
                      edgecolor='#444444')

    if surface_points is not None:
        sp = np.array(surface_points, dtype=float)
        for j, ax in enumerate(axes):
            sc = ax.scatter([], [], c='yellow', s=25, zorder=5,
                            edgecolors='white', linewidths=0.4)
            pt_artists[j] = sc

    def update(frame_num):
        t = frame_indices[frame_num]
        R = R_history[t]

        wf_lines = _ellipsoid_wireframe(a, R)
        for j in range(3):
            for ln, line_pts in zip(wire_artists[j], wf_lines):
                px, py = _project(line_pts, planes[j])
                ln.set_data(px, py)

        axis_scale = a * 1.1
        for j in range(3):
            for k in range(3):
                tip = R[:, k] * axis_scale[k]
                tip_3d = tip.reshape(1, 3)
                orig   = np.zeros((1, 3))
                seg    = np.vstack([orig, tip_3d])
                px, py = _project(seg, planes[j])
                arrow_artists[j][k].set_data(px, py)

        if surface_points is not None:
            sp_rot = sp @ R.T
            for j in range(3):
                px, py = _project(sp_rot, planes[j])
                pt_artists[j].set_offsets(np.c_[px, py])

        angle_deg = np.degrees(
            np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        )
        title_text.set_text(
            f"Ellipsoid rotation — step {int(steps[t])}   "
            f"(cumulative rotation ≈ {angle_deg:.1f}°)"
        )

        artists_out = []
        for j in range(3):
            artists_out += wire_artists[j]
            artists_out += arrow_artists[j]
            if pt_artists[j] is not None:
                artists_out.append(pt_artists[j])
        artists_out.append(title_text)
        return artists_out

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 // fps, blit=True
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    ext = os.path.splitext(output_path)[1].lower()
    if ext == '.gif':
        writer = animation.PillowWriter(fps=fps)
    else:
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800,
                                        extra_args=['-vcodec', 'libx264',
                                                    '-pix_fmt', 'yuv420p'])

    print(f"Saving animation ({n_frames} frames) to '{output_path}'...")
    ani.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"Done. Saved to '{output_path}'")


def plot_rotating_ellipsoid_animation_3d(
    a,
    R_history,
    steps,
    surface_points=None,
    output_path="ellipsoid_rotation_3d.mp4",
    fps=15,
    dpi=120,
    save_every=1,
):
    """
    Animate the rotating ellipsoid with:
      - 1 large 3D wireframe plot on the left
      - 3 stacked 2D projection panels on the right (xy, yz, xz)

    Parameters
    ----------
    a : array-like, shape (3,)
        Semi-axes of the ellipsoid.
    R_history : np.ndarray, shape (T, 3, 3)
        Rotation matrix at each timestep.
    steps : np.ndarray, shape (T,)
        Time values.
    surface_points : list of array-like, optional
        Points on ellipsoid surface in body frame.
    output_path : str
        Path to save MP4 or GIF.
    fps : int
        Frames per second.
    dpi : int
        Resolution.
    save_every : int
        Use every nth timestep as a frame.
    """
    a = np.array(a)
    T = len(steps)
    frame_indices = list(range(0, T, save_every))
    n_frames = len(frame_indices)

    axis_colours = ['tab:red', 'tab:green', 'tab:blue']
    axis_labels  = ['$e_1$', '$e_2$', '$e_3$']
    lim = max(a) * 1.35

    fig = plt.figure(figsize=(13, 8))
    fig.patch.set_facecolor('#0e0e0e')

    ax3d = fig.add_axes([0.01, 0.08, 0.50, 0.84], projection='3d')
    ax3d.set_facecolor('#1a1a1a')
    ax3d.tick_params(colors='white', labelsize=7)
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    ax3d.xaxis.pane.set_edgecolor('#333333')
    ax3d.yaxis.pane.set_edgecolor('#333333')
    ax3d.zaxis.pane.set_edgecolor('#333333')
    ax3d.set_xlim(-lim, lim)
    ax3d.set_ylim(-lim, lim)
    ax3d.set_zlim(-lim, lim)
    ax3d.set_xlabel('x', color='white', fontsize=8)
    ax3d.set_ylabel('y', color='white', fontsize=8)
    ax3d.set_zlabel('z', color='white', fontsize=8)
    ax3d.set_title('3D view', color='white', fontsize=9)

    planes        = ['xy', 'yz', 'xz']
    plane_xlabels = ['x', 'y', 'x']
    plane_ylabels = ['y', 'z', 'z']
    plane_titles  = ['xy plane', 'yz plane', 'xz plane']

    ax2d = []
    for k in range(3):
        top = 0.95 - k * 0.315
        ax  = fig.add_axes([0.56, top - 0.27, 0.41, 0.26])
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white', labelsize=7)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_xlabel(plane_xlabels[k], color='white', fontsize=8)
        ax.set_ylabel(plane_ylabels[k], color='white', fontsize=8)
        ax.set_title(plane_titles[k],   color='white', fontsize=9)
        ax.axhline(0, color='#333333', lw=0.5)
        ax.axvline(0, color='#333333', lw=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')
        ax2d.append(ax)

    title_text = fig.text(0.5, 0.01, "", ha='center', color='white', fontsize=9)

    dummy_lines = _ellipsoid_wireframe(a, np.eye(3))

    wire3d = []
    for lpts in dummy_lines:
        ln, = ax3d.plot(lpts[:, 0], lpts[:, 1], lpts[:, 2],
                        color='#5599ff', lw=0.4, alpha=0.5)
        wire3d.append(ln)

    wire2d = [[] for _ in range(3)]
    for j, ax in enumerate(ax2d):
        for _ in dummy_lines:
            ln, = ax.plot([], [], color='#5599ff', lw=0.5, alpha=0.6)
            wire2d[j].append(ln)

    axes3d_lines = []
    for k in range(3):
        ln, = ax3d.plot([], [], [], color=axis_colours[k],
                        lw=2.0, label=axis_labels[k])
        axes3d_lines.append(ln)
    ax3d.legend(loc='upper left', fontsize=7,
                facecolor='#222222', labelcolor='white', edgecolor='#444444')

    axes2d_lines = [[] for _ in range(3)]
    for j, ax in enumerate(ax2d):
        for k in range(3):
            ln, = ax.plot([], [], color=axis_colours[k], lw=1.8)
            axes2d_lines[j].append(ln)

    sp = np.array(surface_points, dtype=float) if surface_points is not None else None
    sc3d = (ax3d.scatter([], [], [], c='yellow', s=30, zorder=5,
                         edgecolors='white', linewidths=0.4)
            if sp is not None else None)
    sc2d = []
    if sp is not None:
        for ax in ax2d:
            sc2d.append(ax.scatter([], [], c='yellow', s=20, zorder=5,
                                   edgecolors='white', linewidths=0.3))

    def update(frame_num):
        t = frame_indices[frame_num]
        R = R_history[t]
        wf = _ellipsoid_wireframe(a, R)

        for ln, lpts in zip(wire3d, wf):
            ln.set_data(lpts[:, 0], lpts[:, 1])
            ln.set_3d_properties(lpts[:, 2])

        for j in range(3):
            for ln, lpts in zip(wire2d[j], wf):
                px, py = _project(lpts, planes[j])
                ln.set_data(px, py)

        axis_scale = a * 1.05
        for k in range(3):
            tip = R[:, k] * axis_scale[k]
            axes3d_lines[k].set_data([0, tip[0]], [0, tip[1]])
            axes3d_lines[k].set_3d_properties([0, tip[2]])
            seg = np.array([[0, 0, 0], tip])
            for j in range(3):
                px, py = _project(seg, planes[j])
                axes2d_lines[j][k].set_data(px, py)

        if sp is not None:
            sp_rot = sp @ R.T
            sc3d._offsets3d = (sp_rot[:, 0], sp_rot[:, 1], sp_rot[:, 2])
            for j in range(3):
                px, py = _project(sp_rot, planes[j])
                sc2d[j].set_offsets(np.c_[px, py])

        angle_deg = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        title_text.set_text(
            f"t = {steps[t]:.3f}    cumulative rotation ≈ {angle_deg:.1f}°"
        )

        artists = wire3d + axes3d_lines
        for j in range(3):
            artists += wire2d[j] + axes2d_lines[j]
        if sp is not None:
            artists += [sc3d] + sc2d
        artists.append(title_text)
        return artists

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 // fps, blit=True
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    ext = os.path.splitext(output_path)[1].lower()
    if ext == '.gif':
        writer = animation.PillowWriter(fps=fps)
    else:
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800,
                                        extra_args=['-vcodec', 'libx264',
                                                    '-pix_fmt', 'yuv420p'])

    print(f"Saving animation ({n_frames} frames) to '{output_path}'...")
    ani.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"Done. Saved to '{output_path}'")