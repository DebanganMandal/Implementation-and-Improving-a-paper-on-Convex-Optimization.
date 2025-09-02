import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# 1) Simulation parameters
# --------------------------
P = 8                 # number of antennas (array elements)
N = 2000              # number of snapshots/samples
d = 0.5               # spacing in wavelengths (lambda = 1)
theta_sig = 20        # desired DOA (degrees)
theta_int = -40       # interferer DOA (degrees)
SNR_dB = 15           # per-antenna SNR for desired signal (roughly)

rng = np.random.default_rng(0)

# --------------------------
# 2) Helpers
# --------------------------
def steering_vector(P, d, theta_deg):
    """ULA steering vector for angle theta (deg), spacing d (in wavelengths)."""
    k = 2 * np.pi  # 2π/λ with λ=1
    theta = np.deg2rad(theta_deg)
    n = np.arange(P)
    return np.exp(-1j * k * d * n * np.sin(theta))

def qpsk_symbols(n, rng):
    """Unit-power QPSK symbols."""
    const = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    return rng.choice(const, size=n)

# --------------------------
# 3) Generate signals
# --------------------------
# Desired (constant modulus) QPSK
s0 = qpsk_symbols(N, rng)  # desired
# One interferer (complex Gaussian)
s1 = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2)

# Steering vectors
a_sig = steering_vector(P, d, theta_sig)
a_int = steering_vector(P, d, theta_int)

# Noise
sigma = 10**(-SNR_dB/20)  # rough scaling
noise = (rng.standard_normal((P, N)) + 1j * rng.standard_normal((P, N))) * sigma / np.sqrt(2)

# Array data X (P x N), each column is x(t_n)
X = np.outer(a_sig, s0) + np.outer(a_int, s1) + noise

print("X shape (P x N):", X.shape)

# -------------------------------------------------
# 4) Beamforming: y(t) = w^H x(t)
# -------------------------------------------------

# (A) Conventional / Delay-and-Sum beamformer: steer to theta_sig
w_conv = a_sig / np.linalg.norm(a_sig)   # unit-norm weights
y_conv = np.conj(w_conv) @ X             # (1 x N)

# (B) MVDR / Capon beamformer
# Estimate covariance R = E[xx^H] via sample covariance (diagonal loading helps stability)
R = (X @ X.conj().T) / N
# diagonal loading (tiny) to ensure invertibility
eps = 1e-3 * np.trace(R).real / P
R_loaded = R + eps * np.eye(P)

a = a_sig.reshape(-1, 1)
Rinv_a = np.linalg.solve(R_loaded, a)
den = (a.conj().T @ Rinv_a).item()
w_mvdr = (Rinv_a / den).flatten()        # MVDR weights
y_mvdr = np.conj(w_mvdr) @ X             # (1 x N)

print("||w_conv||2=%.3f, ||w_mvdr||2=%.3f" % (np.linalg.norm(w_conv), np.linalg.norm(w_mvdr)))

# -------------------------------------------------
# 5) Quick diagnostics
# -------------------------------------------------

def power_db(z):
    return 10*np.log10(np.mean(np.abs(z)**2) + 1e-12)

# “Raw” single sensor vs beamformed power (rough feel)
p_raw = power_db(X[0])
p_conv = power_db(y_conv)
p_mvdr = power_db(y_mvdr)
print("Avg power (dB): Raw@ant1=%.2f, Conv=%.2f, MVDR=%.2f" % (p_raw, p_conv, p_mvdr))

# Since we know s0, measure how well outputs align to a constant modulus:
# Normalize outputs and check deviation of |y| from 1 after phase de-rotation
# (For QPSK, constant modulus is 1. We only check |y| statistics.)
def modulus_stats(y):
    mag = np.abs(y)
    return np.mean(mag), np.std(mag)

m_conv, s_conv = modulus_stats(y_conv)
m_mvdr, s_mvdr = modulus_stats(y_mvdr)
print("Modulus |y| stats: Conv mean=%.3f std=%.3f, MVDR mean=%.3f std=%.3f" % (m_conv, s_conv, m_mvdr, s_mvdr))

# (Optional) Rough SIR gain estimate (since we know the mixing):
# Project outputs onto the known desired sequence s0
def coherent_snr_like(y, s_ref):
    # “signal” proxy: correlation with s0 (coherent combining)
    num = np.abs(np.vdot(s_ref, y))**2 / len(y)**2
    den = np.mean(np.abs(y)**2) - num
    return 10*np.log10((num + 1e-12) / (den + 1e-12))
sir_conv = coherent_snr_like(y_conv, s0)
sir_mvdr = coherent_snr_like(y_mvdr, s0)
print("Coherent SIR-like (dB): Conv=%.2f, MVDR=%.2f" % (sir_conv, sir_mvdr))

# -------------------------------------------------
# 6) Plots
# -------------------------------------------------

# Constellation before (single antenna) vs after beamforming
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.scatter(np.real(X[0, :300]), np.imag(X[0, :300]), s=6, alpha=0.5)
plt.title("Antenna 1: constellation (first 300)")
plt.xlabel("Re"); plt.ylabel("Im"); plt.axis('equal'); plt.grid(True)

plt.subplot(1,3,2)
plt.scatter(np.real(y_conv[:300]), np.imag(y_conv[:300]), s=6, alpha=0.5)
plt.title("Conventional BF output")
plt.xlabel("Re"); plt.ylabel("Im"); plt.axis('equal'); plt.grid(True)

plt.subplot(1,3,3)
plt.scatter(np.real(y_mvdr[:300]), np.imag(y_mvdr[:300]), s=6, alpha=0.5)
plt.title("MVDR BF output")
plt.xlabel("Re"); plt.ylabel("Im"); plt.axis('equal'); plt.grid(True)

plt.tight_layout()
plt.show()

# Magnitude over time (first 300 samples) — closer to 1 is better for CM signals
plt.figure(figsize=(10,4))
plt.plot(np.abs(X[0,:300]), label='|Antenna 1|')
plt.plot(np.abs(y_conv[:300]), label='|y_conv|')
plt.plot(np.abs(y_mvdr[:300]), label='|y_mvdr|')
plt.axhline(1.0, linestyle='--', linewidth=1, label='CM target')
plt.title('Magnitude (first 300 samples)')
plt.xlabel('n'); plt.ylabel('magnitude')
plt.legend(); plt.grid(True)
plt.show()
