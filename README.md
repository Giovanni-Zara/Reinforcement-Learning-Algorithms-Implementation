# 🤖 Reinforcement Learning & Control Algorithms

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-Environments-black?logo=openai&logoColor=white)](https://gymnasium.farama.org/)

[cite_start]This repository contains a collection of Reinforcement Learning (RL) and Optimal Control algorithms implemented from scratch as part of the **Reinforcement Learning** course at **Sapienza University of Rome (2025/2026)**[cite: 1].

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
<table>
  <tr>
    <td width="70%">
      [cite_start]Implementation of the Policy Iteration algorithm on a modified version of the <strong>FrozenLake</strong> Gymnasium environment[cite: 40].
      <br><br>
      <ul>
        <li><strong>Environment:</strong> Custom <code>FrozenLake</code> grid world[cite: 41, 42].</li>
        <li><strong>Method:</strong> Iterative policy evaluation and policy improvement until convergence[cite: 43].</li>
        <li><strong>Outcome:</strong> The agent learns the optimal path across a slippery grid to reach the goal without falling into holes.</li>
      </ul>
    </td>
    <td width="30%">
      <img src="https://gymnasium.farama.org/_images/frozen_lake.gif" alt="Frozen Lake Environment" width="100%">
    </td>
  </tr>
</table>

### 2. Iterative Linear Quadratic Regulator (iLQR)
<table>
  <tr>
    <td width="70%">
      Implementation of iLQR for solving the <strong>Pendulum</strong> environment[cite: 46]. Unlike standard RL, this is a trajectory optimization method that utilizes knowledge of the system dynamics. 
      <br><br>
      <ul>
        <li><strong>Dynamics:</strong> Implemented the specific equations of motion for the pendulum[cite: 50]:
        $$\dot{\theta}_{t+1} = \dot{\theta}_{t} + (\frac{3g}{2l}\sin \theta + \frac{3}{ml^2}u)dt$$ [cite: 51]
        </li>
        <li><strong>Algorithm:</strong>
          <ul>
             <li><strong>Backward Pass:</strong> Computed $Q$ and $V$ derivatives using Riccati equations to find feedback gains $K$ and $k$[cite: 54]. This included terms from the cost's 2nd order Taylor expansion[cite: 55].</li>
             <li><strong>Forward Pass:</strong> Updated the control sequence $u$ using the computed gains: $u_{new} = k + K(x_{new} - x_{old})$[cite: 59, 60].</li>
          </ul>
        </li>
      </ul>
    </td>
    <td width="30%">
      <img src="https://gymnasium.farama.org/_images/pendulum.gif" alt="Pendulum Environment" width="100%">
    </td>
  </tr>
</table>

---

## Assignment 2: SARSA(λ) & Linear Function Approximation
**Focus:** Temporal Difference Learning & Feature Engineering

### 1. SARSA(λ) (Taxi-v3)
<table>
  <tr>
    <td width="70%">
      Implementation of the on-policy TD control algorithm with eligibility traces[cite: 333]. 
      <br><br>
      <ul>
        <li><strong>Environment:</strong> Taxi-v3 (Discrete state space)[cite: 333].</li>
        <li><strong>Key Concept:</strong> <code>Eligibility Traces</code> (decaying memory of past states) to speed up convergence compared to 1-step SARSA.</li>
        <li><strong>Result:</strong> Efficient navigation for passenger pickup and drop-off.</li>
      </ul>
    </td>
    <td width="30%">
      <img src="./taxi.gif" alt="Taxi Environment" width="100%">
    </td>
  </tr>
</table>

### 2. Linear Q-Learning with RBF (Mountain Car)
<table>
  <tr>
    <td width="70%">
      Solving a continuous state space problem using Linear Function Approximation[cite: 337].
      <br><br>
      <ul>
        <li><strong>Environment:</strong> MountainCar-v0 (Underpowered car requiring momentum)[cite: 338].</li>
        <li><strong>Feature Extraction:</strong> Implemented a <strong>Radial Basis Function (RBF)</strong> encoder to project the 2D state space (position, velocity) into a higher-dimensional feature space[cite: 337, 346].</li>
        <li><strong>Update Rule:</strong> Q-Learning utilizing the gradient of the linear approximation weights[cite: 337].</li>
      </ul>
    </td>
    <td width="30%">
      <img src="https://gymnasium.farama.org/_images/mountain_car.gif" alt="Mountain Car Environment" width="100%">
    </td>
  </tr>
</table>

---

## Assignment 3: Trust Region Policy Optimization (TRPO)
**Focus:** Deep Reinforcement Learning & Continuous Control

<table>
  <tr>
    <td width="70%">
      A complete implementation of <strong>TRPO</strong> to solve the challenging <strong>CarRacing-v2</strong> environment from pixel inputs[cite: 137, 159]. This project was built using <strong>PyTorch</strong>[cite: 152].
      <br><br>
      The agent learns to steer, accelerate, and brake (discrete action space) by processing raw $96 \times 96$ RGB images[cite: 160, 162].
    </td>
    <td width="30%">
      <img src="https://gymnasium.farama.org/_images/car_racing.gif" alt="Car Racing Environment" width="100%">
    </td>
  </tr>
</table>

### 🏗️ Architecture (Actor-Critic)
[cite_start]The model utilizes a shared Convolutional Neural Network (CNN) feature extractor feeding into separate Actor and Critic heads[cite: 194, 215]. 

| Component | Specification |
|:--- |:--- |
| **Input** | [cite_start]$96 \times 96 \times 3$ RGB Images [cite: 160] |
| **Feature Extractor** | [cite_start]3 Conv Layers (Stride 4, 2, 1) + ReLU activations [cite: 218] |
| **Actor Head** | [cite_start]Fully Connected (256 $\to$ 128 $\to$ 5 discrete actions) [cite: 208, 211] |
| **Critic Head** | [cite_start]Fully Connected (256 $\to$ 128 $\to$ 1 scalar value) [cite: 210, 212] |

### ⚙️ Algorithmic Details
* [cite_start]**Constraint:** Updates are constrained within a "Trust Region" (KL Divergence $\le \delta$) to ensure monotonic improvement and stability[cite: 172, 177].
* **Optimization:**
    * [cite_start]**Actor:** Updated using **Conjugate Gradient** to approximate the inverse Fisher Information Matrix ($F^{-1}g$) without explicitly forming the matrix[cite: 189].
    * [cite_start]**Critic:** Updated using standard Adam optimization on Mean Squared Error[cite: 261, 265].
* [cite_start]**Advantage Estimation:** Uses **Generalized Advantage Estimation (GAE)** to balance bias and variance[cite: 233].

### 🛠️ Environment Enhancements
* [cite_start]**Zoom Skip:** Skipped the first 50 frames of "zoom-in" animation to stabilize training[cite: 268].
* [cite_start]**Stuck Detection:** Implemented a penalty and early reset if the car produces negative rewards for >100 steps[cite: 270, 271].
* [cite_start]**Preprocessing:** Frame normalization ($0-1$) and transposing to channel-first format $(C, H, W)$[cite: 274].

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/GiovanniZara/RL-Algorithms-Implementation.git](https://github.com/GiovanniZara/RL-Algorithms-Implementation.git)
   cd RL-Algorithms-Implementation
   ```

2. **Install dependencies:**
   ```bash
   cd <ur desired folder>
   pip install -r requirements.txt
   ```

3. **Run specific assignments:**
   *Example: To evaluate the TRPO agent:*
   ```bash
   cd car_racing
   python main.py --evaluate
   ```
