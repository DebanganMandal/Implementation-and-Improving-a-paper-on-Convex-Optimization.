import numpy as np
import cvxpy as cp

def solve_cma_trace_norm(X, solver="SCS"):
    """
    Solve the convex CMA relaxation:
        minimize (1/N) * sum_n (x_n^H W x_n - 1)^2 + tr(W)
        subject to W >> 0 (Hermitian PSD)
    Inputs:
        X : (P, N) complex-ndarray, columns are snapshots x_n
    Returns:
        W_hat : (P, P) Hermitian PSD ndarray
        w_hat : (P,) principal eigenvector of W_hat (unit-norm)
    """
    P, N = X.shape

    W = cp.Variable((P, P), hermitian=True)

    residuals = []
    for n in range(N):
        xn = X[:, n]
        residuals.append(cp.real(cp.quad_form(xn, W)) - 1)

    obj = (1.0/N) * cp.sum_squares(cp.hstack(residuals)) + cp.trace(W)
    constraints = [W >> 0]

    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=solver, verbose=False)

    if W.value is None:
        raise RuntimeError("Optimization did not converge.")

    W_hat = W.value
    evals, evecs = np.linalg.eigh(W_hat)
    w_hat = evecs[:, np.argmax(evals)]
    w_hat = w_hat / np.linalg.norm(w_hat)
    return W_hat, w_hat

# Generate synthetic narrowband snapshots
rng = np.random.default_rng(0)

P = 8            # sensors
N = 250          # snapshots
num_interf = 3

# Steering vectors for a ULA (half-wavelength spacing)
def steering(P, angle_deg):
    d = 0.5  # lambda/2 spacing
    k = 2*np.pi  # wavenumber normalized by lambda
    angle = np.deg2rad(angle_deg)
    phase = k * d * np.arange(P) * np.sin(angle)
    return np.exp(1j*phase) / np.sqrt(P)

a_sig = steering(P, 20)                   # desired direction
A_int = np.column_stack([steering(P, ang) for ang in (-45, -15, 40)[:num_interf]])

# QPSK desired signal and complex-Gaussian interferers
s = np.exp(1j * (np.pi/2) * rng.integers(0, 4, size=N))  # unit-modulus CM
I = (rng.normal(size=(num_interf, N)) + 1j*rng.normal(size=(num_interf, N))) / np.sqrt(2)

sigma2 = 0.1
noise = (rng.normal(size=(P, N)) + 1j*rng.normal(size=(P, N))) * np.sqrt(sigma2/2)

X = a_sig[:, None] * s[None, :] + A_int @ I + noise  # P x N snapshots


# Solve (pick either version)
W_hat, w_hat = solve_cma_trace_norm(X, solver="SCS")
# W_hat, w_hat = solve_cma_trace_norm_with_z(X, solver="SCS")

print("Principal eigenvalue of W_hat:", np.max(np.linalg.eigvalsh(W_hat)).real)
print("The required beamforming vector:", w_hat)
print("||w_hat||:", np.linalg.norm(w_hat))
# Beamformer output snapshots:
y = np.conj(w_hat) @ X
print("Mean |y|^2 (should be ~1):", np.mean(np.abs(y)**2))
