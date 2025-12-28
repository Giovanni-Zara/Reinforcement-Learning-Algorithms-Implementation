# 🤖 Reinforcement Learning & Control Algorithms

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-Environments-black?logo=openai&logoColor=white)](https://gymnasium.farama.org/)

This repository contains a collection of Reinforcement Learning (RL) and Optimal Control algorithms implemented from scratch as part of the **Reinforcement Learning** course at **Sapienza University of Rome (2025/2026)**.

The projects progress from classical tabular methods and optimal control to advanced Deep Reinforcement Learning with continuous action spaces.

---

## 📚 Table of Contents
- [Assignment 1: Policy Iteration & iLQR](#assignment-1-policy-iteration--ilqr)
- [Assignment 2: SARSA(λ) & Linear Function Approximation](#assignment-2-sarsa%CE%BB--linear-function-approximation)
- [Assignment 3: Trust Region Policy Optimization (TRPO)](#assignment-3-trust-region-policy-optimization-trpo)
- [Installation](#installation)

---

## Assignment 1: Policy Iteration & iLQR
**Focus:** Dynamic Programming & Optimal Control

### 1. Policy Iteration (FrozenLake)
<p align="center">
  <img src="https://gymnasium.farama.org/_images/frozen_lake.gif" alt="Frozen Lake Environment" width="600"/>
</p>
Implementation of the Policy Iteration algorithm on a modified version of the **FrozenLake** Gymnasium environment. 
* **Environment:** Custom `FrozenLake` grid world.
* **Method:** Iterative policy evaluation and policy improvement until convergence.
* **Outcome:** The agent learns the optimal path across a slippery grid to reach the goal without falling into holes.

### 2. Iterative Linear Quadratic Regulator (iLQR)
<p align="center">
  <img src="https://gymnasium.farama.org/_images/pendulum.gif" alt="Pendulum Environment" width="600"/>
</p>
Implementation of iLQR for solving the **Pendulum** environment. unlike standard RL, this is a trajectory optimization method that utilizes knowledge of the system dynamics.
* **Dynamics:** Implemented the specific equations of motion for the pendulum: 
    $$\dot{\theta}_{t+1} = \dot{\theta}_{t} + (\frac{3g}{2l}\sin \theta + \frac{3}{ml^2}u)dt$$
* **Algorithm:** * **Backward Pass:** Computed $Q$ and $V$ derivatives using the Riccati equations to find feedback gains $K$ and $k$.
    * **Forward Pass:** Updated the control sequence $u$ using the computed gains: $u_{new} = k + K(x_{new} - x_{old})$.

---

## Assignment 2: SARSA(λ) & Linear Function Approximation
**Focus:** Temporal Difference Learning & Feature Engineering

### 1. SARSA(λ) (Taxi-v3)
<p align="center">
  <img src="https://gymnasium.farama.org/_images/taxi.gif" alt="Taxi Environment" width="600"/>
</p>
Implementation of the on-policy TD control algorithm with eligibility traces.
* **Environment:** Taxi-v3 (Discrete state space).
* **Key Concept:** `Eligibility Traces` (decaying memory of past states) to speed up convergence compared to 1-step SARSA.
* **Result:** Efficient navigation for passenger pickup and drop-off.

### 2. Linear Q-Learning with RBF (Mountain Car)
<p align="center">
  <img src="https://gymnasium.farama.org/_images/mountain_car.gif" alt="Mountain Car Environment" width="600"/>
</p>
Solving a continuous state space problem using Linear Function Approximation.
* **Environment:** MountainCar-v0 (Underpowered car requiring momentum).
* **Feature Extraction:** Implemented a **Radial Basis Function (RBF)** encoder to project the 2D state space (position, velocity) into a higher-dimensional feature space.
* **Update Rule:** Q-Learning utilizing the gradient of the linear approximation weights.

---

## Assignment 3: Trust Region Policy Optimization (TRPO)
**Focus:** Deep Reinforcement Learning & Continuous Control

<p align="center">
  <img src="https://gymnasium.farama.org/_images/car_racing.gif" alt="Car Racing Environment" width="600"/>
</p>

A complete implementation of **TRPO** to solve the challenging **CarRacing-v2** environment from pixel inputs. This project was built using **PyTorch**.

### 🏗️ Architecture (Actor-Critic)
The model utilizes a shared Convolutional Neural Network (CNN) feature extractor feeding into separate Actor and Critic heads.

| Component | Specification |
|:--- |:--- |
| **Input** | $96 \times 96 \times 3$ RGB Images |
| **Feature Extractor** | 3 Conv Layers (Stride 4, 2, 1) + ReLU activations |
| **Actor Head** | Fully Connected (256 $\to$ 128 $\to$ 5 discrete actions) |
| **Critic Head** | Fully Connected (256 $\to$ 128 $\to$ 1 scalar value) |

### ⚙️ Algorithmic Details
* **Constraint:** Updates are constrained within a "Trust Region" (KL Divergence $\le \delta$) to ensure monotonic improvement and stability.
* **Optimization:** * **Actor:** Updated using **Conjugate Gradient** to approximate the inverse Fisher Information Matrix ($F^{-1}g$) without explicitly forming the matrix (Hessian-vector products).
    * **Critic:** Updated using standard Adam optimization on Mean Squared Error.
* **Advantage Estimation:** Uses **Generalized Advantage Estimation (GAE)** to balance bias and variance.

### 🛠️ Environment Enhancements
* **Zoom Skip:** Skipped the first 50 frames of "zoom-in" animation to stabilize training.
* **Stuck Detection:** Implemented a penalty and early reset if the car produces negative rewards for >100 steps.
* **Preprocessing:** Frame normalization ($0-1$) and transposing to channel-first format $(C, H, W)$.

---

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/RL-Algorithms-Implementation.git](https://github.com/your-username/RL-Algorithms-Implementation.git)
   cd RL-Algorithms-Implementation
