import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm 
import pandas as pd


tau = 0.05
T = 1
dt = 0.001

noise_amp = 0.5
steps = 1000000

def drift(A, tau, T):
    expA = expm(tau*A)
    C = expA @ expA.T #in theory it should be C_tau = exp(tau @ A) @ exp(tau @ A).T
    C += 1e-8 * np.eye(3)
    C_inv = np.linalg.inv(C)
    trA2 = np.trace(A @ A)
    trCinv = np.trace(C_inv)
    drift = -A@A + (trA2/trCinv)*C_inv -(trCinv/(3*T))*A
    return drift

def noise(noise_amp, dt):
    dW = np.random.randn(3,3)
    dW -= np.trace(dW)/3 * np.eye(3)
    return noise_amp *np.sqrt(dt)*dW

def rk2_sde(A, dt, tau, T, noise_amp):
    dW = noise(noise_amp, dt)
    fA = drift(A, tau, T)
    A_1 = A + fA*dt + dW
    A_1 -= np.trace(A_1) / 3 * np.eye(3)
    fA_1 = drift(A_1, tau, T)
    A_next = A + 1/2 * (fA + fA_1) * dt + dW
    #A_next -= np.trace(A_next)/3 * np.eye(3) #by including this line, it removes isotropic-ness
    return A_next

np.random.seed(7)

A_history = np.zeros((steps, 3, 3))

A1 = np.random.randn(3,3) * 0.1
# A1 -= np.trace(A1)/3 * np.eye(3)      #by including this line, it removes isotropic-ness

for i in range(steps):
    A1 = rk2_sde(A1, dt, tau, T, noise_amp)
    A_history[i] = A1

t = np.arange(steps) * dt

plt.plot(t, A_history[:,0,0], label="A11")
plt.plot(t, A_history[:,1,1], label="A22")
plt.plot(t, A_history[:,2,2], label="A33")
plt.legend()
plt.xlabel("time")
plt.ylabel("A_ij")
plt.title("Diagonal comp")
plt.show()

plt.plot(t, A_history[:,0,1], label="A12")
plt.plot(t, A_history[:,0,2], label="A13")
plt.plot(t, A_history[:,1,2], label="A23")
plt.legend()
plt.xlabel("time")
plt.ylabel("A_ij")
plt.title("Non diagonal comp")
plt.show()


data = np.zeros((steps, 10))  # 1 time column + 9 matrix entries
data[:, 0] = t
data[:, 1] = A_history[:, 0, 0]
data[:, 2] = A_history[:, 0, 1]
data[:, 3] = A_history[:, 0, 2]
data[:, 4] = A_history[:, 1, 0]
data[:, 5] = A_history[:, 1, 1]
data[:, 6] = A_history[:, 1, 2]
data[:, 7] = A_history[:, 2, 0]
data[:, 8] = A_history[:, 2, 1]
data[:, 9] = A_history[:, 2, 2]

# Create a DataFrame
df = pd.DataFrame(data, columns=[
    "time", "A11", "A12", "A13",
    "A21", "A22", "A23",
    "A31", "A32", "A33"
])

# Save to CSV
df.to_csv("grad_u.csv", index=False)
print("Saved grad_u.csv")