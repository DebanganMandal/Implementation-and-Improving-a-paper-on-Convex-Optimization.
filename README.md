# Constant Modulus Beamforming via Convex Optimization

This repository contains our implementation and extension of the paper:

**Adler, A., & Wax, M. (2017). Constant Modulus Beamforming via Convex Optimization.**  
[arXiv:1704.03004](https://arxiv.org/abs/1704.03004)

---

## 📖 Project Overview
The classical Constant Modulus Algorithm (CMA) for blind beamforming is formulated as a **non-convex quartic optimization problem**, making it prone to local minima.  
The referenced paper introduces a **convex reformulation** using matrix lifting and nuclear norm relaxation, ensuring global optimality and enabling additional linear constraints.

In this project, we:
- Implement the convex CMA and LCCMA formulations in **Python (CVXPY)**.  
- Generate synthetic array snapshots (QPSK desired signal, Gaussian interferers, noise) to test the approach.  
- Evaluate beamforming performance in terms of output power and interference rejection.  
- Explore possible improvements and extensions to the convex relaxation.

---

## 📌 Course Context
This work is part of a term project for
CRL734: Optimization, IIT Delhi
Instructor: Professor Dr. Akaash Arora
