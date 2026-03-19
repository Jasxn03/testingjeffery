import numpy as np
from jeffery4 import Ellipsoid

from load_data            import load_grad_u_csv
from stress_functions     import compute_stress_from_A, compute_stress_timeseries
from stress_plot          import plot_stress_timeseries, plot_traction_timeseries
from orientation          import plot_rotation_history, plot_cumulative_vs_net_rotation
from orientation          import plot_rotating_ellipsoid_animation, plot_rotating_ellipsoid_animation_3d
from surface_stress_plot  import plot_stress_colormap_timeseries, plot_stress_colormap, plot_stress_colormap_2
from stress_colormap      import plot_stress_colormap_timeseries_fast, animate_stress_colormap, plot_stress_colormap_snapshot, plot_stress_colormap_snapshot_2


# ============================================================================
# Configuration
# ============================================================================

a  = np.array([2.0, 1.0, 1.0])
mu = 1.5

surface_points = [
    [2.0,  0.0,  0.0],    # tip along x1-axis
    [0.0,  1.0,  0.0],    # tip along x2-axis
    [0.0,  0.0,  1.0],    # tip along x3-axis
]

# ============================================================================
# Single-step example
# ============================================================================

A = np.array([[ 0.10,  0.15,  0.02],
              [-0.05,  0.20,  0.08],
              [ 0.03, -0.06,  0.15]])

results, epsilon, omega = compute_stress_from_A(
    Ellipsoid, a, A, mu, surface_points
)

# ============================================================================
# Time series
# ============================================================================

steps, A_timeseries = load_grad_u_csv("grad_u.csv")
T = len(steps)

ts_results, rotation_history = compute_stress_timeseries(
    Ellipsoid, a, mu, A_timeseries, surface_points, steps=steps, track_rotation=True
)

# Extract traction time series at surface point 0 -> shape (T, 3)
traction_ts = np.array([ts_results[t][0]['traction'] for t in range(T)])
print(f"\nTraction time series shape: {traction_ts.shape}")
print(f"Mean traction at point 0:   {traction_ts.mean(axis=0)}")
print(f"Std  traction at point 0:   {traction_ts.std(axis=0)}")

# Extract stress tensor time series at surface point 1 -> shape (T, 3, 3)
sigma_ts = np.array([ts_results[t][1]['sigma'] for t in range(T)])
print(f"\nStress tensor time series shape: {sigma_ts.shape}")

# ============================================================================
# Plots
# ============================================================================

plot_stress_timeseries(steps, ts_results, surface_points, output_dir="stress_plots")

plot_traction_timeseries(steps, ts_results, surface_points, output_dir="stress_plots")

if rotation_history is not None:
    plot_rotation_history(steps, rotation_history, output_dir="stress_plots")
    plot_cumulative_vs_net_rotation(steps, rotation_history, output_dir="stress_plots")
    print(f"\nFinal cumulative rotation: {np.degrees(rotation_history['angle'][-1]):.2f} degrees")
    print(f"Mean rotation axis: {rotation_history['axis'].mean(axis=0)}")

plot_stress_colormap(
    Ellipsoid, a, mu, A_timeseries[0], R=rotation_history['R'][0],
    n_theta=80, n_phi=80,
    output_dir="stress_plots",
    timestep=0
)

plot_stress_colormap_2(
    Ellipsoid, a, mu, A_timeseries[0], R=rotation_history['R'][0],
    n_theta=800, n_phi=800, # it is too long with 800, 80 is relatively quick
    output_dir="stress_plots",
    timestep=0
)

# plot_rotating_ellipsoid_animation(
#     a=a,
#     R_history=rotation_history['R'],
#     steps=steps,
#     surface_points=surface_points,
#     output_path="stress_plots/ellipsoid_rotation.mp4",
#     fps=60,
#     save_every=10,
# )

# plot_rotating_ellipsoid_animation_3d(
#     a=a,
#     R_history=rotation_history['R'],
#     steps=steps,
#     surface_points=surface_points,
#     output_path="stress_plots/ellipsoid_rotation_3d.mp4",
#     fps=60,
#     save_every=10,
# )

# plot_stress_colormap_timeseries_fast(
#     ellipsoid_class=Ellipsoid,
#     a=a,
#     mu=mu,
#     steps=steps,
#     A_timeseries=A_timeseries,
#     R_history=rotation_history['R'],
#     n_theta=50, n_phi=50,
#     output_dir="stress_colormaps",
#     save_every=500,
#     consistent_colorbar=True,
# )

# animate_stress_colormap(
#     ellipsoid_class=Ellipsoid,
#     a=a,
#     mu=mu,
#     steps=steps,
#     A_timeseries=A_timeseries,
#     R_history=rotation_history['R'],
#     n_theta=40,
#     n_phi=40,
#     output_path="stress_animation.mp4",
#     fps=60,
#     cmap='plasma',
#     consistent_colorbar=True,
# )