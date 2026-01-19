# Learning Fourier Neural Operator

This repository reproduces and analyzes the Fourier Neural Operator (FNO) for learning solution operators of partial differential equations.

We focus on the 1D viscous Burgers' equation at Reynolds number Re=10 and study generalization across spatial resolutions.

---

## Background

Fourier Neural Operators (Li et al., 2020) learn mappings between infinite-dimensional function spaces and have shown strong performance on parametric PDEs.

This project investigates:
- operator learning for nonlinear PDEs
- spectral neural architectures
- resolution invariance and generalization
- data efficiency of operator learning

---

## PDE Setup

We solve the viscous Burgers' equation:

$$
u_t + u u_x = \nu u_{xx}; \nu = Re=10.
$$

The model learns the operator:

$$
\mathcal{G}: u(x,0) \to u(x,T)
$$

<img width="1678" height="1302" alt="64resolution" src="https://github.com/user-attachments/assets/0233e7bb-7a72-4ef2-bf06-8da1a0c2787e" />

<img width="1678" height="1302" alt="256resolution" src="https://github.com/user-attachments/assets/b34dda6e-1611-4420-86fa-27b447b8cfc4" />

<img width="1678" height="1302" alt="1024resolution" src="https://github.com/user-attachments/assets/8211264a-a058-40fa-bdeb-51aa0a4a1579" />

<img width="3034" height="1113" alt="FNO_Berger_Learning" src="https://github.com/user-attachments/assets/e9ded1cd-8b0e-4512-ad22-26d08df4e17c" />

We observe stable operator generalization across discretizations. Also for datasets use: https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-

## Research Theme
-- Operator learning

-- Scientific machine learning

-- Spectral neural networks

-- PDE surrogate modeling

-- Physics-informed learning
