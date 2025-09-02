import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(123)

# ------------------------
# 1) Simulate input data
# ------------------------
P = 8    # antennas
N = 2000 # samples

# Desired QPSK signal (constant modulus)
rng = np.random.default_rng(0)
symbols = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
s0 = rng.choice(symbols, size=N)

# One interferer
s1 = (rng.standard_normal(N) + 1j*rng.standard_normal(N)) / np.sqrt(2)

# Steering vectors
def steering_vector(P, d, theta_deg):
    k = 2*np.pi
    theta = np.deg2rad(theta_deg)
    return np.exp(-1j * k * 0.5 * np.arange(P) * np.sin(theta))

def cost_function(y):
    return torch.mean((torch.abs(y.squeeze())**2 - 1)**2)

a_sig = steering_vector(P, 0.5, 20)  # desired at 20°
a_int = steering_vector(P, 0.5, -40) # interferer at -40°

# Noise
sigma = 0.2
noise = (rng.standard_normal((P, N)) + 1j*rng.standard_normal((P, N))) * sigma/np.sqrt(2)

# Array data: X (P x N)
X_np = np.outer(a_sig, s0) + np.outer(a_int, s1) + noise
X = torch.tensor(X_np, dtype=torch.cfloat)  # for PyTorch

# ------------------------
# 2) Beamformer NN
# ------------------------
# class Beamformer(nn.Module):
#     def __init__(self, P):
#         super().__init__()
#         self.w = nn.Parameter(torch.randn(P, dtype=torch.cfloat))  # learnable weights

#     def forward(self, X):
#         # X: (P x N), w: (P)
#         return torch.conj(self.w).resolve_conj().unsqueeze(0) @ X   # (1 x N)

class BeamformerReal(nn.Module):
    def __init__(self, P):
        super().__init__()
        self.wr = nn.Parameter(torch.randn(P, dtype=torch.float32))
        self.wi = nn.Parameter(torch.randn(P, dtype=torch.float32))

    def forward(self, X):  # X: (P,N) complex
        # w^H = (wr - j wi)^T
        wH_real = self.wr.unsqueeze(0)         # (1,P)
        wH_imag = -self.wi.unsqueeze(0)        # (1,P)
        # (wH_real + j wH_imag) @ (X_real + j X_imag)
        Xr, Xi = X.real, X.imag
        y_real = wH_real @ Xr - wH_imag @ Xi   # (1,N)
        y_imag = wH_real @ Xi + wH_imag @ Xr   # (1,N)
        return torch.complex(y_real, y_imag)   # (1,N)


# model = Beamformer(P)
model = BeamformerReal(P)
optimizer = optim.Adam(model.parameters(), lr=0.05)

# ------------------------
# 3) Train with constant modulus loss
# ------------------------

loss_history = []


for epoch in range(1000):
    optimizer.zero_grad()
    y = model(X)         # (1 x N)
    loss = cost_function(y)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss = {loss.item():.4f}")

# Final beamformer output
y_out = model(X).detach().numpy().flatten()

# ------------------------
# 4) Plots
# ------------------------

# plt.figure(figsize=(12,4))

# # (a) Constellation of raw input (antenna 1)
# plt.subplot(1,2,1)
# plt.scatter(np.real(X_np[0,:300]), np.imag(X_np[0,:300]), s=6, alpha=0.6)
# plt.title("Raw Antenna 1: constellation")
# plt.xlabel("Re"); plt.ylabel("Im"); plt.axis("equal"); plt.grid(True)

# # (b) Constellation after beamforming (NN output)
# plt.subplot(1,2,2)
# plt.scatter(np.real(y_out[:300]), np.imag(y_out[:300]), s=6, alpha=0.6, c='orange')
# plt.title("NN Beamformer output: constellation")
# plt.xlabel("Re"); plt.ylabel("Im"); plt.axis("equal"); plt.grid(True)

# plt.tight_layout()
# plt.show()

# # Magnitude over time (first 200 samples)
# plt.figure(figsize=(10,4))
# plt.plot(np.abs(X_np[0,:200]), label="|Antenna 1|")
# plt.plot(np.abs(y_out[:200]), label="|Beamformer output|", linewidth=2)
# plt.axhline(1.0, linestyle="--", color="k", label="Target modulus = 1")
# plt.title("Magnitude over time")
# plt.xlabel("Sample index"); plt.ylabel("Magnitude")
# plt.legend(); plt.grid(True)
# plt.show()

plt.figure(figsize=(8,4))
plt.plot(loss_history, label="Training loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Beamformer Training Loss Curve")
plt.grid(True)
plt.legend()
plt.show()

w_real = model.wr.detach().cpu().numpy()
w_imag = model.wi.detach().cpu().numpy()
w_complex = w_real + 1j * w_imag
w_learned = w_complex/np.linalg.norm(np.abs(w_complex))

theta_sig = 20
a_sig = steering_vector(P, 0.5, theta_sig)
a_sig = a_sig / np.linalg.norm(a_sig)

print("Learned weight vector (complex):")
print(w_complex)

# Optional: print magnitude and phase separately
print("\nMagnitudes:", np.abs(w_complex))
print("Phases (degrees):", np.rad2deg(np.angle(w_complex)))

similarity = np.abs(np.vdot(a_sig, w_learned))
print("Similarity:", similarity)