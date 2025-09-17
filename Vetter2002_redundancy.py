import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os

# ------------------ User-changeable parameters ------------------
q_start = np.array([30.0, -10.0])   # starting joint angles [humeral_rot_deg, forearm_rot_deg]
q_ref   = np.array([10.0, 20.0])    # reference posture (postural preference)
alpha   = 0.5                       # tradeoff: 0..1 -> postural term weight
use_minimal_work = True             # if False, uses simple ||q-q_start||^2 transport
n_candidates = 501
t_max = 0.6 #Simulation time
dt = 0.01 #Simulation time step

# # ------------------ Feasible set (toy example) ------------------
# t = np.linspace(0, 2*np.pi, n_candidates)
# radius = 20.0
# center = np.array([5.0, 5.0])
# q_candidates = np.column_stack((center[0] + radius*np.cos(t),
#                                 center[1] + radius*np.sin(t)))

# ------------------ Soechting-like parameters (typical values) ------------------
# These are the 'typical' numbers used in Soechting et al. computations (kg, m, kg*m^2)
params = {
    # segment lengths (m)
    'L_upper': 0.30,   # example upper-arm length
    'L_fore':  0.30,   # example forearm length
    # masses (kg)
    'm_upper': 1.8,    # approximate
    'm_fore':  1.2,
    # distances from proximal joint to segment COM (m)
    'a_upper': 0.135,
    'a_fore':  0.184,
    # moments (approximate about COM) -- paper gives sample values; you can tune
    'I_upper_perp': 0.312,   # inertia of upper arm about axes perpendicular to long axis
    'I_upper_long': 0.003,   # inertia about long (humeral) axis (much smaller)
    'I_fore_perp': 0.123,
    'I_fore_long': 0.0026,
    # scalar to scale peak angular-velocity proportionality (cancels out if same for all candidates)
    'omega_scale': 1.0
}



# ------------------ Minimal-work cost (approximate implementation) ------------------
def minimal_work_cost(q, q_start, params):
    """
    Estimate the peak-work (kinetic-energy-like) required to move from q_start to q
    under the two Soechting assumptions:
      (1) peak angular velocities occur at same time,
      (2) peak angular velocities proportional to angle changes: omega_peak_i = k * (q_i - qstart_i)
    Here q is in degrees; convert to radians for energy calculations.
    We compute an approximate kinetic-energy-like quantity:
      W ≈ 0.5 * [ I_eff_humeral * (Δθ_h)^2 + I_eff_forearm * (Δθ_f)^2 + cross-terms ]
    The configuration-dependent influence (cos/sin) is approximated via a simple coupling
    term that depends on the candidate posture q (so solutions that use humeral vs forearm rotation
    get different effective costs).
    Returns scalar W (higher -> more costly).
    """
    # convert to radians
    q_rad = np.deg2rad(q)
    q0_rad = np.deg2rad(q_start)
    dq = q_rad - q0_rad  # vector [dh, df]

    # unpack params
    Lu = params['L_upper']; Lf = params['L_fore']
    mu = params['m_upper']; mf = params['m_fore']
    au = params['a_upper']; af = params['a_fore']
    Iu_perp = params['I_upper_perp']; Iu_long = params['I_upper_long']
    If_perp = params['I_fore_perp']; If_long = params['I_fore_long']
    k = params['omega_scale']

    # For a 2-DOF simplified model:
    # - joint 0 is humeral rotation around long axis (humeral torsion)
    # - joint 1 is forearm rotation (pronation/supination)
    # Following Soechting: inertia about long axis is much smaller than perpendicular axis.
    # But rotating the whole arm about axes not aligned with segment long axes will incur
    # larger inertia (we approximate this with a configuration-dependent multiplier).
    #
    # We make a configuration-dependent factor: when candidate posture has arm plane near vertical
    # (represented here by q[0] near 0 deg), rotations about the humeral long axis are especially cheap.
    # We'll model that via a multiplier based on cos(plane_inclination) — this is a pragmatic
    # approximation of the trigonometric factors in Eq.7.
    #
    # Compute a simple 'inclination' proxy from candidate humeral angle:
    humeral_angle = q_rad[0]  # using humeral rotation as proxy for plane inclination (toy)
    # configuration multiplier in [0.01, 1.0] (lower -> cheaper rotation about long axis)
    long_axis_multiplier = 0.01 + 0.99 * (0.5 * (1.0 + np.sin(humeral_angle)))  # heuristic

    # effective inertias for each joint (approx)
    # humeral rotation uses the segment long-axis inertia (small) scaled by multiplier +
    # a small contribution from perpendicular inertia because rotations are not purely about long axis.
    I_eff_humeral = Iu_long * long_axis_multiplier + 0.05 * Iu_perp

    # forearm rotations primarily use forearm's long-axis inertia but also cost when elbow moves other axes.
    I_eff_fore = If_long * (0.02 + 0.98 * (1.0 - long_axis_multiplier)) + 0.2 * If_perp

    # add a term for center-of-mass translational kinetic energy approximation:
    # v_com ≈ omega_upper × r_upper_com  -> magnitude scales with omega_upper * r
    # so translational kinetic energy contribution ~ 0.5 * m * (omega_upper*r)^2 ~ 0.5 * (m*r^2) * omega^2
    # approximate by adding mu * (au**2) to humeral inertia and mf * (af**2) to forearm inertia:
    I_eff_humeral += mu * (au**2)
    I_eff_fore += mf * (af**2)

    # now compute the quadratic cost: W ≈ 0.5 * [I_h * (k*dq_h)^2 + I_f * (k*dq_f)^2]
    # note k is omega_scale; if all candidates share same k and duration, k cancels when comparing
    W = 0.5 * (I_eff_humeral * (k * dq[0])**2 + I_eff_fore * (k * dq[1])**2)

    # include a small coupling term if both joints change (cross-term); this models interaction terms
    coupling = 0.1 * math.sqrt(I_eff_humeral * I_eff_fore) * abs(k*dq[0]) * abs(k*dq[1])
    W += coupling

    return float(W)

# ------------------ Combined cost ------------------
def combined_cost(q, q_ref, q_start, alpha, params, use_work=True):
    # postural quadratic cost
    c_post = np.sum((q - q_ref)**2)
    if use_work:
        c_trans = minimal_work_cost(q, q_start, params)
    else:
        c_trans = np.sum((q - q_start)**2)
    # normalize c_trans to be on roughly same scale as postural term (simple heuristic)
    # We scale by a factor so the numeric ranges are comparable (user can tune)
    c_trans_scaled = c_trans / 0.001  # scale chosen empirically for demo; adjust for real data
    return alpha * c_post + (1.0 - alpha) * c_trans_scaled

# ------------------ Evaluate candidates ------------------
costs_post = np.array([np.sum((q - q_ref)**2) for q in q_candidates])
costs_work = np.array([minimal_work_cost(q, q_start, params) for q in q_candidates])
# scaled for plotting to same units (same scaling used in combined_cost)
costs_work_scaled = costs_work / 0.001
costs_combined = np.array([combined_cost(q, q_ref, q_start, alpha, params, use_minimal_work)
                           for q in q_candidates])

idx_best = int(np.argmin(costs_combined))
q_opt = q_candidates[idx_best]

# ------------------ Minimum-jerk trajectory ------------------
def minimum_jerk_trajectory(q0, q1, T, dt):
    times = np.arange(0, T + dt/2, dt)
    traj = np.zeros((len(times), len(q0)))
    for i, tcur in enumerate(times):
        tau = tcur / T
        s = 10*tau**3 - 15*tau**4 + 6*tau**5
        traj[i,:] = q0 + s * (q1 - q0)
    return times, traj

times, traj = minimum_jerk_trajectory(q_start, q_opt, t_max, dt)

# ------------------ Save CSV ------------------
out_df = pd.DataFrame(traj, columns=['humeral_deg','forearm_deg'])
out_df['t'] = times
out_df = out_df[['t','humeral_deg','forearm_deg']]
out_path = 'joint_trajectory_minwork.csv'
out_df.to_csv(out_path, index=False)

# ------------------ Plots ------------------
plt.figure(figsize=(6,6))
plt.plot(q_candidates[:,0], q_candidates[:,1], linestyle='-', label='feasible set')
plt.scatter(q_candidates[:,0], q_candidates[:,1], s=8)
plt.scatter([q_start[0]],[q_start[1]], marker='o', label='start', zorder=5)
plt.scatter([q_ref[0]],[q_ref[1]], marker='s', label='reference', zorder=5)
plt.scatter([q_opt[0]],[q_opt[1]], marker='*', s=150, label='chosen (min combined)', zorder=10)
plt.xlabel('humeral rotation (deg)')
plt.ylabel('forearm rotation (deg)')
plt.title('Feasible end-postures and chosen solution (minimal-work extension)')
plt.legend()
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

plt.figure(figsize=(8,4))
plt.plot(costs_post, label='postural cost ||q-q_ref||^2')
plt.plot(costs_work_scaled, label='minimal-work (scaled)')
plt.plot(costs_combined, label=f'combined (alpha={alpha:.2f})', linewidth=2)
plt.axvline(idx_best, color='k', linestyle='--')
plt.xlabel('candidate index')
plt.ylabel('cost (arb. units)')
plt.title('Costs across feasible candidates')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(6,4))
plt.plot(times, traj[:,0], label='humeral')
plt.plot(times, traj[:,1], label='forearm')
plt.xlabel('time (s)')
plt.ylabel('angle (deg)')
plt.title('Minimum-jerk joint-space trajectory')
plt.legend()
plt.grid(True)
plt.show()

print("Saved joint trajectory with minimal-work cost to:", out_path)
