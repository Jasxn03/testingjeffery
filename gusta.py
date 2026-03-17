import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from orientation import integrate_orientation

CSV_PATH   = "grad_u.csv"
LW_CSV_PATH  = "grad_u_LW.csv" 

def load_csv(path):
    df     = pd.read_csv(path)
    times  = df['time'].values
    A_series = np.zeros((len(df), 3, 3))
    for i in range(3):
        for j in range(3):
            A_series[:, i, j] = df[f'A{i+1}{j+1}'].values
    print(f"Loaded {len(times)} timesteps  ({times[0]:.4f} → {times[-1]:.4f})")
    return times, A_series


# # ── Integrate orientation ────────────────────────────────
# print("\nIntegrating orientation ODE...")
# R_history, omega_history, angle_history, axis_history, dtheta_history = \
#     integrate_orientation(AXES, A_series, times)
# # Digitised from Gustavsson et al. 2014 Fig. 2 (right), DNS curve.
# # x = lambda (their aspect ratio, same as our r = a/b)
# # y = <ndot^2> * tau_K^2  (normalised squared tumbling rate)
# # Disks: lambda < 1,  Rods: lambda > 1,  Sphere: lambda = 1
# _GUSTAVSSON_LAMBDA = np.array([
#     0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00,
#     1.10, 1.30, 1.50, 2.00, 2.50, 3.00, 4.00, 5.00, 7.00, 10.0
# ])
# _GUSTAVSSON_NDOT2 = np.array([
#     0.75, 0.68, 0.62, 0.55, 0.50, 0.47, 0.44, 0.42, 0.41, 0.40, 0.40,
#     0.41, 0.42, 0.44, 0.47, 0.49, 0.50, 0.52, 0.53, 0.55, 0.57
# ])


def compute_tumbling_rate_vs_r(A_series, times, aspect_ratios):
    """
    For each aspect ratio, integrate orientation and compute
    <omega^2> * tau_K^2  (normalised squared tumbling rate).

    tau_K = 1 / sqrt(Tr<A^T A>) — Kolmogorov time from data.

    Returns dict: r -> mean_ndot2_normalised
    """
    from orientation import integrate_orientation

    # Kolmogorov time from the full A timeseries
    AtA      = np.einsum('tji,tjk->tik', A_series, A_series)   # A^T A
    TrAtA    = np.einsum('tii->t', AtA)                         # Tr(A^T A)
    tau_K    = 1.0 / np.sqrt(np.mean(TrAtA))
    print(f"  Kolmogorov time tau_K = {tau_K:.4f}")

    results = {}
    for r in aspect_ratios:
        axes = [float(r), 1.0, 1.0]
        print(f"  Integrating r={r}...", end=" ", flush=True)
        R_h, omega_h, *_ = integrate_orientation(axes, A_series, times)

        # omega_h is body-frame angular velocity (T, 3)
        ndot2 = np.mean(np.sum(omega_h**2, axis=1))   # <|omega|^2>
        ndot2_norm = ndot2 * tau_K**2                  # normalised

        results[r] = {
            'ndot2':      ndot2,
            'ndot2_norm': ndot2_norm,
        }
        print(f"<ndot^2>={ndot2:.4f}  normalised={ndot2_norm:.4f}")

    return results, tau_K


# def plot_step5_gustavsson_comparison(tumble_results, tau_K, aspect_ratios):
#     """
#     Compare our <ndot^2> * tau_K^2 vs lambda curve to Gustavsson Fig. 2 (right).
#     """
#     rs       = np.array(aspect_ratios)
#     our_vals = np.array([tumble_results[r]['ndot2_norm'] for r in rs])

#     fig, ax = plt.subplots(figsize=(8, 6))

#     # Gustavsson DNS curve
#     ax.plot(_GUSTAVSSON_LAMBDA, _GUSTAVSSON_NDOT2,
#             'o--', color='steelblue', lw=2, ms=5, label='Gustavsson 2014 (DNS)')

#     # Our curve
#     ax.plot(rs, our_vals,
#             's-', color='darkorange', lw=2.5, ms=7,
#             label='Chevillard-Meneveau model (my stuff)')

#     ax.axvline(1.0, color='gray', lw=0.8, linestyle=':')
#     ax.set_xlabel('Aspect ratio $\\lambda = a/b$', fontsize=12)
#     ax.set_ylabel('$\\langle \\omega^2 \\rangle \\tau_K^2$', fontsize=13)
#     ax.set_title('Normalised tumbling rate vs aspect ratio\n'
#                  'Comparison with Gustavsson et al. (2014) Fig. 2',
#                  fontsize=12, fontweight='bold')
#     ax.legend(fontsize=8)
#     ax.grid(True, alpha=0.3)
#     ax.set_xscale('log')

#     # Annotate disk/rod regions
#     ax.text(0.25, ax.get_ylim()[1]*0.95, 'Disks\n$(\\lambda < 1)$',
#             ha='center', fontsize=9, color='gray')
#     ax.text(3.0,  ax.get_ylim()[1]*0.95, 'Rods\n$(\\lambda > 1)$',
#             ha='center', fontsize=9, color='gray')

#     fig.tight_layout()
#     fig.savefig(OUTDIR + 'step5_gustavsson_comparison.png', dpi=150, bbox_inches='tight')
#     plt.close(fig)
#     print("Saved step5_gustavsson_comparison.png")

_GUSTAVSSON_LAMBDA = np.array([
    0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00,
    1.10, 1.30, 1.50, 2.00, 2.50, 3.00, 4.00, 5.00, 7.00, 10.0
])
_GUSTAVSSON_NDOT2 = np.array([
    0.75, 0.68, 0.62, 0.55, 0.50, 0.47, 0.44, 0.42, 0.41, 0.40, 0.40,
    0.41, 0.42, 0.44, 0.47, 0.49, 0.50, 0.52, 0.53, 0.55, 0.57
])

def plot_comparison_two_datasets(
    results_dns,
    results_lw,
    aspect_ratios
):
    rs = np.array(aspect_ratios)

    dns_vals = np.array([results_dns[r]['ndot2_norm'] for r in rs])
    lw_vals  = np.array([results_lw[r]['ndot2_norm'] for r in rs])

    fig, ax = plt.subplots(figsize=(8, 6))

    # Gustavsson DNS reference
    ax.plot(_GUSTAVSSON_LAMBDA, _GUSTAVSSON_NDOT2,
            'o--', color='steelblue', lw=2, ms=5,
            label='Gustavsson 2014 (DNS ref)')

    # Your DNS (from grad_u.csv)
    ax.plot(rs, dns_vals,
            's-', color='black', lw=2.5, ms=6,
            label='Your DNS (grad_u.csv)')

    # LW model
    ax.plot(rs, lw_vals,
            'd-', color='darkorange', lw=2.5, ms=6,
            label='LW model (grad_u_LW.csv)')

    ax.axvline(1.0, color='gray', lw=0.8, linestyle=':')

    ax.set_xlabel('Aspect ratio $\\lambda = a/b$', fontsize=12)
    ax.set_ylabel('$\\langle \\omega^2 \\rangle \\tau_K^2$', fontsize=13)
    ax.set_title('Tumbling rate comparison', fontsize=12, fontweight='bold')

    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    fig.tight_layout()
    plt.show()


aspect_ratios = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]

# --- DNS data ---
times_dns, A_dns = load_csv(CSV_PATH)
results_dns, tau_dns = compute_tumbling_rate_vs_r(
    A_dns, times_dns, aspect_ratios
)

# --- LW model ---
times_lw, A_lw = load_csv(LW_CSV_PATH)
results_lw, tau_lw = compute_tumbling_rate_vs_r(
    A_lw, times_lw, aspect_ratios
)


# if RUN_STEP5:
#     print("\n── Step 5: Tumbling rate vs aspect ratio (Gustavsson comparison) ──")
#     # No traction needed here — only omega from orientation integration
#     ASPECT_RATIOS_SWEEP = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
#     tumble_results, tau_K = compute_tumbling_rate_vs_r(
#         A_b, times_b, ASPECT_RATIOS_SWEEP)
#     plot_step5_gustavsson_comparison(tumble_results, tau_K, ASPECT_RATIOS_SWEEP)


plot_comparison_two_datasets(
    results_dns,
    results_lw,
    aspect_ratios
)