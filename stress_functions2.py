import numpy as np
import jax.numpy as jnp


def outward_normal(x, a):
    """
    Compute the outward unit normal to the ellipsoid surface at point x.

    For ellipsoid (x1/a1)^2 + (x2/a2)^2 + (x3/a3)^2 = 1,
    the unnormalized normal is [x1/a1^2, x2/a2^2, x3/a3^2].

    Parameters
    ----------
    x : array-like, shape (3,)
        Point on the ellipsoid surface.
    a : array-like, shape (3,)
        Semi-axes of the ellipsoid [a1, a2, a3].

    Returns
    -------
    n : jnp.ndarray, shape (3,)
        Outward unit normal vector at x.
    """
    x = jnp.array(x)
    a = jnp.array(a)
    n = x / (a ** 2)
    return n / jnp.linalg.norm(n)


def compute_surface_stress(ellipsoid, x_surface, a, mu):
    """
    Compute the stress tensor and traction vector at a point on the
    ellipsoid surface using Jeffery's solution.

    Parameters
    ----------
    ellipsoid : Ellipsoid
        An Ellipsoid object (already initialised with epsilon, omega, mu).
    x_surface : array-like, shape (3,)
        A point on the ellipsoid surface.
    a : array-like, shape (3,)
        Semi-axes of the ellipsoid [a1, a2, a3].
    mu : float
        Dynamic viscosity of the fluid.

    Returns
    -------
    sigma : jnp.ndarray, shape (3, 3)
        Stress tensor at x_surface.
    traction : jnp.ndarray, shape (3,)
        Traction vector (force per unit area) at x_surface.
    eps_local : jnp.ndarray, shape (3, 3)
        Local strain rate tensor at x_surface.
    n : jnp.ndarray, shape (3,)
        Outward unit normal at x_surface.
    """
    x_surface = jnp.array(x_surface)

    # du_dx = ellipsoid.du_dx(x_surface).reshape(3, 3)
    p = ellipsoid.p(x_surface)

    # eps_local = 0.5 * (du_dx + du_dx.T)

    # I = jnp.eye(3)
    # sigma = -p * I + 2 * mu * eps_local

    sigma = ellipsoid.sigma(x_surface)

    n = outward_normal(x_surface, a)
    traction = sigma @ n
    eps_local = (sigma + p * np.eye(3)) / (2 * mu)

    return sigma, traction, eps_local, n


def compute_stress_from_A(ellipsoid_class, a, A, mu, x_surface_points):
    """
    Full pipeline: given a stochastic velocity gradient A, decompose it
    into strain rate and rotation, build the ellipsoid, and compute the
    stress/traction at each surface point.

    Parameters
    ----------
    ellipsoid_class : class
        The Ellipsoid class to instantiate.
    a : array-like, shape (3,)
        Semi-axes of the ellipsoid [a1, a2, a3].
    A : np.ndarray, shape (3, 3)
        Velocity gradient tensor (e.g. from a stochastic model).
    mu : float
        Dynamic viscosity of the fluid.
    x_surface_points : list of array-like, each shape (3,)
        Points on the ellipsoid surface to evaluate stress at.

    Returns
    -------
    results : list of dict
        Each dict contains 'x', 'sigma', 'traction', 'eps_local', 'normal'.
    epsilon : np.ndarray, shape (3, 3)
        Strain rate tensor derived from A.
    omega : np.ndarray, shape (3,)
        Rotation vector derived from A.
    """
    A = np.array(A)

    epsilon = 0.5 * (A + A.T)
    Omega   = 0.5 * (A - A.T)
    omega   = np.array([Omega[2, 1], Omega[0, 2], Omega[1, 0]])

    print("=" * 60)
    print("STRESS COMPUTATION FROM VELOCITY GRADIENT A")
    print("=" * 60)

    print("\nInput velocity gradient A:")
    for row in A:
        print(f"  [{row[0]:8.4f}  {row[1]:8.4f}  {row[2]:8.4f}]")

    print("\nStrain rate tensor epsilon = 0.5*(A + A.T):")
    for row in epsilon:
        print(f"  [{row[0]:8.4f}  {row[1]:8.4f}  {row[2]:8.4f}]")

    print(f"\nRotation vector omega = [{omega[0]:.4f}, {omega[1]:.4f}, {omega[2]:.4f}]")

    ellipsoid = ellipsoid_class(a, epsilon, mu=mu)
    print(f"\nEllipsoid created with semi-axes a = {a}")

    results = []
    print(f"\nComputing stress at {len(x_surface_points)} surface point(s)...\n")

    for i, x in enumerate(x_surface_points):
        x = jnp.array(x)
        sigma, traction, eps_local, n = compute_surface_stress(ellipsoid, x, a, mu)

        results.append({
            'x':         x,
            'sigma':     sigma,
            'traction':  traction,
            'eps_local': eps_local,
            'normal':    n,
        })

        print(f"  Point {i+1}: x = [{x[0]:.3f}, {x[1]:.3f}, {x[2]:.3f}]")
        print(f"    Normal n        = [{n[0]:.4f}, {n[1]:.4f}, {n[2]:.4f}]")
        print(f"    Traction t      = [{traction[0]:.6f}, {traction[1]:.6f}, {traction[2]:.6f}]")
        print(f"    |traction|      = {jnp.linalg.norm(traction):.6f}")
        print(f"    Stress tensor sigma:")
        for row in sigma:
            print(f"      [{row[0]:10.6f}  {row[1]:10.6f}  {row[2]:10.6f}]")
        print()

    return results, epsilon, omega


def precompute_transfer_matrices(ellipsoid, x_surface_points):
    """
    Precompute transfer matrices for du_dx and p at each surface point.

    This only needs to be done ONCE — the matrices depend only on ellipsoid
    geometry, not on the flow condition (epsilon, omega).

    Parameters
    ----------
    ellipsoid : Ellipsoid
        Any Ellipsoid instance with the correct geometry.
    x_surface_points : list of array-like, each shape (3,)
        Points on the ellipsoid surface.

    Returns
    -------
    transfer_data : list of dict
        Each dict contains:
          'x'       - the surface point
          'n'       - outward unit normal, shape (3,)
          'T_dudx'  - transfer matrix for du_dx, shape (9, 12)
          'T_p'     - transfer matrix for p,     shape (12,)
    """
    transfer_data = []
    for x in x_surface_points:
        x = jnp.array(x)
        transfer_data.append({
            'x':      x,
            'n':      outward_normal(x, ellipsoid.a),
            'T_dudx': ellipsoid.transfer(x, field='du_dx'),  # shape (9, 12)
            'T_p':    ellipsoid.transfer(x, field='p'),      # shape (12,)
        })
    return transfer_data


def A_to_mode(A, ellipsoid):
    """
    Decompose a velocity gradient A into (epsilon, omega) and pack
    into the mode vector expected by the transfer matrices.

    Parameters
    ----------
    A : np.ndarray, shape (3, 3)
        Velocity gradient tensor at one timestep.
    ellipsoid : Ellipsoid
        Ellipsoid instance (used for its pack() method).

    Returns
    -------
    mode : jnp.ndarray, shape (12,)
        Packed mode vector [epsilon (9) | omega (3)].
    """
    A       = np.array(A)
    epsilon = 0.5 * (A + A.T)
    Omega   = 0.5 * (A - A.T)
    omega   = np.array([Omega[2, 1], Omega[0, 2], Omega[1, 0]])
    return ellipsoid.pack(epsilon)


def compute_stress_timeseries(ellipsoid_class, a, mu, A_timeseries, x_surface_points, steps=None, track_rotation=True):
    """
    Efficiently compute stress and traction at multiple surface points
    for every timestep in a time series of velocity gradient tensors.

    Transfer matrices are computed ONCE and reused for all timesteps,
    making this far faster than rebuilding the ellipsoid each step.

    Parameters
    ----------
    ellipsoid_class : class
        The Ellipsoid class to instantiate.
    a : array-like, shape (3,)
        Semi-axes of the ellipsoid [a1, a2, a3].
    mu : float
        Dynamic viscosity of the fluid.
    A_timeseries : np.ndarray, shape (T, 3, 3)
        Time series of velocity gradient tensors, T timesteps.
    x_surface_points : list of array-like, each shape (3,)
        Points on the ellipsoid surface to evaluate stress at.

    Returns
    -------
    results : list of lists, shape (T, N_points)
        results[t][i] is a dict with keys 'sigma', 'traction', 'eps_local'
        at timestep t and surface point i.

    Example
    -------
    results = compute_stress_timeseries(Ellipsoid, a, mu, A_ts, surface_pts)

    # Extract traction time series at surface point 0
    traction_ts = np.array([results[t][0]['traction'] for t in range(T)])
    # shape: (T, 3)

    # Extract full stress tensor time series at surface point 1
    sigma_ts = np.array([results[t][1]['sigma'] for t in range(T)])
    # shape: (T, 3, 3)
    """
    from orientation import integrate_orientation

    A_timeseries = np.array(A_timeseries)
    T = len(A_timeseries)
    N = len(x_surface_points)
    I = jnp.eye(3)

    # --- Step 1: Build ellipsoid ONCE using dummy flow condition.
    #             The flow condition does not affect the transfer matrices —
    #             only the geometry (a, mu) matters here. ---
    dummy_eps   = np.eye(3)
    ellipsoid   = ellipsoid_class(a, dummy_eps, mu=mu)
    ellipsoid.use_surface_mode()

    # --- Step 2: Precompute transfer matrices at all surface points (once) ---
    print(f"Precomputing transfer matrices for {N} surface point(s)... ", end="")
    transfer_data = precompute_transfer_matrices(ellipsoid, x_surface_points)
    print("done.")

    # -- rotation bit --
    rotation_history = None
    if track_rotation and steps is not None:
        print("Integration orientation ODE...")
        R_history, omega_history, angle_history, axis_history, dtheta_history = integrate_orientation(a, A_timeseries, steps)
        rotation_history = {
            'R': R_history,
            'omega': omega_history,
            'angle': angle_history,
            'axis': axis_history,
            'dtheta': dtheta_history,
        }
        print("done.")
    else:
        R_history = np.tile(np.eye(3), (T, 1, 1))

    # --- Step 3: Loop over timesteps — only cheap matrix multiplies ---
    print(f"Computing stress for {T} timesteps x {N} surface points...")

    results = [[None] * N for _ in range(T)]

    import time
    t0 = time.time()
    for t, A_t in enumerate(A_timeseries[:1000]):
        R_t    = R_history[t]
        A_body = R_t.T @ A_t @ R_t
        epsilon = 0.5 * (A_body + A_body.T)
        ellipsoid.set_strain(epsilon)
        ellipsoid.set_coefs()
        for i, td in enumerate(transfer_data):
            sigma     = ellipsoid.sigma(td['x'])
            p_t       = -np.trace(sigma) / 3
            traction  = sigma @ td['n']
            eps_local = (sigma + p_t * I) / (2 * mu)
            results[t][i] = {
                'sigma':     sigma,
                'traction':  traction,
                'eps_local': eps_local,
            }
    print(f"1000 steps took {time.time()-t0:.2f}s")

    for t, A_t in enumerate(A_timeseries):
        R_t    = R_history[t]
        A_body = R_t.T @ A_t @ R_t

        epsilon = 0.5 * (A_body + A_body.T)
        ellipsoid.set_strain(epsilon)
        ellipsoid.set_coefs()

        for i, td in enumerate(transfer_data):
            sigma    = ellipsoid.sigma(td['x'])
            p_t      = -np.trace(sigma)/3
            traction = sigma @ td['n']
            eps_local = (sigma + p_t * I) / (2 * mu)

            results[t][i] = {
                'sigma':     sigma,
                'traction':  traction,
                'eps_local': eps_local,
            }

    print("Done.")
    return results, rotation_history