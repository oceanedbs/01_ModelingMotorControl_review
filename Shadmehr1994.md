# Article : Adaptive Representation of Dynamics during Learning of a Motor Task, from Reza Shadmehr and Ferdinand0 A. Mussa-lvaldi


This project simulates the dynamics and control of a two-link planar arm (shoulder + elbow), based on a simplified biomechanical model.
The simulation includes minimum-jerk trajectory generation, inverse kinematics, and torque-based control with stiffness/viscosity feedback.
It can be run with or without an external force field applied to the end-effector.

---

## 📖 Overview

The arm is modeled as a **two-link manipulator** in 2D space:

- Link 1: Upper arm (`L1`, `m1`, `I1`, `r1`)
- Link 2: Forearm (`L2`, `m2`, `I2`, `r2`)

The control system is inspired by Shadmehr & Mussa-Ivaldi’s motor control framework:

\[
\tau = I(q) \, \ddot{q}_{des} + G(q, \dot{q}) + \hat{E}(q, \dot{q}) - K (q - q_{des}) - V (\dot{q} - \dot{q}_{des})
\]

where:
- \( I(q) \): inertia matrix  
- \( G(q,\dot{q}) \): Coriolis/centrifugal terms  
- \( K, V \): stiffness and viscosity feedback  
- \( q_{des}, \dot{q}_{des}, \ddot{q}_{des} \): desired joint states from inverse kinematics  
- \( \hat{E} \): external/environmental forces (optional)

The trajectory is generated using a **minimum-jerk trajectory planner** in Cartesian space, then mapped to joint space via **inverse kinematics**.

---

## 📂 File Structure

- `main.py` → the provided code (simulation, controller, plotting)
- `TwoDLArm_func.py` → helper functions:
  - `direct_kinematics(q, L1, L2)`
  - `inverse_kinematics(x, y, L1, L2)`
  - `jacobian(q, L1, L2)`
  - `jacobian_dot(q, dq, L1, L2)`
  - `inertia_matrix(...)`, `coriolis_matrix(...)`
  - `fun_minjerktrajectory(...)` (trajectory generator)
  - `endpoint_force_field(...)` (optional external disturbance)
  - `plot_arm(...)` (visualize arm configuration)

---

## ⚙️ Key Components

### 1. Arm Parameters
Defined at the top of the code (masses, lengths, inertias, centers of mass).  
Stiffness and viscosity matrices \(K, V\) define joint feedback.

### 2. Trajectory Generation
- Minimum-jerk trajectory is computed from initial → final hand position.
- Desired **Cartesian position, velocity, acceleration** → converted into **joint states** using inverse kinematics + Jacobian.

### 3. Controller
Implements torque computation. It interpolates desired states at the current simulation time and applies viscoelastic feedback.

\[
\tau = I(q) \, \ddot{q}_{des} + G(q, \dot{q}) + \hat{E}(q, \dot{q}) - K (q - q_{des}) - V (\dot{q} - \dot{q}_{des})
\]


### 4. System Dynamics
Forward dynamics computed as:

\[
\ddot{q} = I(q)^{-1} \, (C - G(q,\dot{q}) - E)
\]

with Euler integration updating states \((q, \dot{q})\).

### 5. Simulation Loop
- Builds desired trajectory (`desTraj` DataFrame).  
- Runs forward dynamics step by step.  
- Supports:
  - **Free movement** (`E_func=None`)  
  - **Force-field disturbance** (`E_func=endpoint_force_field`)

### 6. Visualization
- Desired trajectory (Cartesian position, velocity, joint angles).  
- Simulated hand path in workspace.  
- Arm initial/final positions.

---

## 🧩 Block Diagram

![Block Diagram of the Two-Link Arm Control System](img/Shadmehr1994.png)

## ▶️ How to Run

1. Install requirements:
   ```bash
   pip install numpy scipy matplotlib pandas

2. Ensure TwoDLArm_func.py is in the same directory.

3. Run the script:
    ```bash
    python main.py


This will:
- Plot the desired trajectory.
- Simulate movement without force field.
- Simulate movement with Cartesian force field.
- Plot resulting hand trajectories.