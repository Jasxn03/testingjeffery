import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm 
import pandas as pd
from scipy.stats import gaussian_kde, skew
from sklearn.mixture import GaussianMixture

from load_data import load_grad_u_csv

steps, A_timeseries = load_grad_u_csv('grad_u.csv')

KDE = np.empty((3,3), dtype = object)
for i in range(3):
    for j in range(3):
        KDE[i,j] = gaussian_kde(A_timeseries[:,i,j])


GMM = np.empty((3,3), dtype = object)
weights = np.empty((3,3), dtype = object)
means = np.empty((3,3), dtype = object)
variances = np.empty((3,3), dtype = object)
for i in range(3):
    for j in range(3):
        gmm = GaussianMixture(n_components=3)
        gmm.fit(A_timeseries[:,i,j].reshape(-1,1))
        x_vals = np.linspace(-5, 5, 300).reshape(-1,1)
        GMM[i,j] = np.exp(gmm.score_samples(x_vals))
        weights[i,j] = gmm.weights_
        means[i,j] = gmm.means_.flatten()
        variances[i,j] = gmm.covariances_.flatten()


X_VALS = np.linspace(-5, 5, 300)

fig, axes = plt.subplots(3,3, figsize=(10,10))
for i in range(3):
    for j in range(3):
        ax = axes[i,j]
        ax.plot(X_VALS, KDE[i,j](X_VALS), label='KDE')
        ax.plot(X_VALS, GMM[i,j], '--', label='GMM')
        ax.set_title(f'A[{i},{j}]')
        ax.legend()
plt.tight_layout()
plt.show()


rows = []
for i in range(3):
    for j in range(3):
        for k in range(3): 
            rows.append(
                {
                    'component': f'A_{i}{j}',
                    'weight': weights[i,j][k],
                    'mean' : means[i,j][k],
                    'variance': variances[i,j][k]
                }
            )

df = pd.DataFrame(rows)

df.to_csv("gmm_coefficients.csv", index=False)
print("Saved gmm_coefficients.csv")
