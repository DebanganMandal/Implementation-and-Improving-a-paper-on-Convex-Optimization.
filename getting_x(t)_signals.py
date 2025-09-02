import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Parameters
# --------------------------
P = 8                 # number of antennas
N = 1000              # number of time samples
d = 0.5               # spacing between antennas (in wavelengths)
theta_sig = 20        # angle of desired signal (degrees)
theta_int = -40       # angle of interferer (degrees)
SNR_dB = 20           # signal-to-noise ratio in dB

# --------------------------
# Helper functions
# --------------------------
def steering_vector(P, d, theta_deg):
    """Create steering vector for a ULA with P elements, spacing d, and angle theta."""
    k = 2 * np.pi  # since wavelength λ = 1 unit
    theta = np.deg2rad(theta_deg)
    return np.exp(-1j * k * d * np.arange(P) * np.sin(theta))

# --------------------------
# Generate signals
# --------------------------
# Desired QPSK signal (constant modulus)
symbols = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
s0 = np.random.choice(symbols, size=N)

# Interferer (Gaussian)
s1 = (np.random.randn(N) + 1j*np.random.randn(N)) / np.sqrt(2)

# Noise
sigma = 10**(-SNR_dB/20)   # convert SNR to noise power
noise = (np.random.randn(P, N) + 1j*np.random.randn(P, N)) * sigma / np.sqrt(2)

# --------------------------
# Construct received signal x(t)
# --------------------------
a_sig = steering_vector(P, d, theta_sig)
a_int = steering_vector(P, d, theta_int)

# Each column of X is x(t_n)
X = np.outer(a_sig, s0) + np.outer(a_int, s1) + noise

print("Shape of X (P x N):", X.shape)

# --------------------------
# Plot real part of first antenna’s signal
# --------------------------
plt.figure(figsize=(8,4))
plt.plot(np.real(X[0,:50]), label="Real part (antenna 1)")
plt.title("Received signal at antenna 1 (first 50 samples)")
plt.xlabel("Sample index")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()
