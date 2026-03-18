# extract kde estimate from csv
# sample function

import pandas as pd
import numpy as np


import pandas as pd
import numpy as np

csv_path = 'gmm_coefficients.csv'

def sample_A_tensor_batch(csv_path, n_rows=3, n_samples=1000):
    df = pd.read_csv(csv_path)
    components = sorted(df["component"].unique())
    
    n_comp = len(components)
    samples = np.empty((n_comp, n_samples))
    
    for idx, comp in enumerate(components):
        group = df[df["component"] == comp]
        
        weights = group["weight"].values
        means = group["mean"].values
        stds = np.sqrt(group["variance"].values)
        
        ks = np.random.choice(len(weights), size=n_samples, p=weights)
        samples[idx, :] = np.random.normal(means[ks], stds[ks])
    
    n_cols = n_comp // n_rows
    return samples.reshape(n_rows, n_cols, n_samples)

