import numpy as np
import pandas as pd


def load_grad_u_csv(filepath):
    """
    Load a CSV of velocity gradient time series with columns:
        step, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y, w_z

    Each row is one timestep. The 9 gradient components are interpreted as:
        row 0 of A: [u_x, u_y, u_z]   (gradient of u-velocity)
        row 1 of A: [v_x, v_y, v_z]   (gradient of v-velocity)
        row 2 of A: [w_x, w_y, w_z]   (gradient of w-velocity)

    so that A[i,j] = d(velocity_i)/d(x_j).

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    steps : np.ndarray, shape (T,)
        Step indices from the CSV.
    A_timeseries : np.ndarray, shape (T, 3, 3)
        Velocity gradient tensor at each timestep.
    """
    df = pd.read_csv(filepath)

    # Expect exactly these columns (case-insensitive)
    expected = ['time', 'a11', 'a12', 'a13', 'a21', 'a22', 'a23', 'a31', 'a32', 'a33']
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}\n"
                         f"Found columns: {list(df.columns)}")

    steps = df['time'].to_numpy()

    # Stack the 9 components into (T, 3, 3)
    A_timeseries = np.stack([
        df[['a11', 'a12', 'a13']].to_numpy(),   # row 0
        df[['a21', 'a22', 'a23']].to_numpy(),   # row 1
        df[['a31', 'a32', 'a33']].to_numpy(),   # row 2
    ], axis=1)  # shape (T, 3, 3)

    print(f"Loaded {len(steps)} timesteps from '{filepath}'")
    return steps, A_timeseries