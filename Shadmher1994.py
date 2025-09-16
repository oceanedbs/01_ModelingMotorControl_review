from multiprocessing.resource_sharer import stop
import numpy as np
import math
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from TwoDLArm_func import *
import pandas as pd
# --- Paramètres mécaniques du bras (Table 1 de l'article) ---
# Longueurs (m), masses (kg), inerties (kg·m²), centres de masse (m)

L1 = 0.3  # Longueur bras
L2 = 0.33  # Longueur avant-bras
m1 = 1.93  # Masse bras (parametres anthropometriques)
m2 = 1.52  # Masse avant-bras
I1 = 0.0141  # Inertie bras
I2 = 0.0188  # Inertie avant-bras
r1 = 0.165  # Centre de masse bras
r2 = 0.19   # Centre de masse avant-bras

t_max = 0.5

#Initial and final positions
x0, y0 = 0, 0  # Position initiale
xf, yf = 0, 0.1  # Position finale (10 cm à 0°)

vx0, vy0 = 0, 0  # Vitesse initiale
vxf, vyf = 0, 0  # Vitesse finale

acx0, acy0 = 0, 0  # Accélération initiale
acxf, acyf = 0, 0  # Accélération finale

cond_init = np.array([np.deg2rad(45), np.deg2rad(90), 0, 0]) 

# Matrices de stiffness et viscosité (N·m/rad et N·m·s/rad)
K = np.array([[-15, -6], [-6, -16]])  # Stiffness
V = np.array([[-2.3, -0.9], [-0.9, -2.4]])  # Viscosité

desTraj = pd.DataFrame()

handPos = direct_kinematics((cond_init[0], cond_init[1]), L1, L2)

# --- Contrôleur (Eq. 9) ---
def controller(q, dq, t, desTraj,
                L1, L2, m1, m2, r1, r2, I1, I2,
                K, V, E_hat=None):
    print("controller")
    """
    Returns tau = C(q,dq,t) following Shadmehr Eq.9 form:
    tau = I(q) @ ddq_des + G(q,dq) + E_hat - K (q - q_des) - V (dq - dq_des)

    desTraj: pandas DataFrame with columns time, trajX, trajY, velX, velY, accelX, accelY
    E_hat: either None (== zero) or a callable E_hat(q,dq) returning joint torques (2,)
    """
    # clamp time to avoid integrator overshoot
    t_clamped = float(np.clip(t, desTraj["time"].iloc[0], desTraj["time"].iloc[-1]))

    # # Desired joint states by interpolation
    q_des = np.array([
        np.interp(t_clamped, desTraj["time"], desTraj["q1_des"]),
        np.interp(t_clamped, desTraj["time"], desTraj["q2_des"])
    ])
    dq_des = np.array([
        np.interp(t_clamped, desTraj["time"], desTraj["dq1_des"]),
        np.interp(t_clamped, desTraj["time"], desTraj["dq2_des"])
    ])
    ddq_des = np.array([
        np.interp(t_clamped, desTraj["time"], desTraj["ddq1_des"]),
        np.interp(t_clamped, desTraj["time"], desTraj["ddq2_des"])
    ])
    
    print("q_des:", q_des, "dq_des:", dq_des)

    # Modèle interne du bras (D_hat)
    I = inertia_matrix(q, L1, L2, m1, m2, r1, r2, I1, I2)
    G = coriolis_matrix(q, dq, L1, L2, m2, r2)
    D_hat = I @ ddq_des + G

    # Modèle de l'environnement (E_hat)
    if E_hat is None:
        E_hat = np.zeros(2)  # Pas de modèle initial
    else:
        E_hat = E_hat(q, dq)  # Champ de forces appris

    # Feedback viscoélastique (S)
    S = K @ (q - q_des) + V @ (dq - dq_des)

    # Contrôleur complet (Eq. 9)
    # C = D_hat + E_hat - S
    C = S
    return C

# --- Dynamique complète (Eq. 4) ---
def dynamics(y, t, desTraj, K, V,
             L1, L2, m1, m2, r1, r2, I1, I2,
             E_func=None, E_hat=None):
    print("dynamics")
    """Équations différentielles pour la dynamique du bras."""
    q, dq = y[:2], y[2:] #récupérer q et dq initiaux
 

    # External environment forces
    if E_func is None:
        E = np.zeros(2)
    else:
        if E_func.__name__ == "endpoint_force_field":
            J = jacobian(q, L1, L2)
            x_dot = J @ dq
            f = E_func(x_dot)
            E = J.T @ f
        else:
            E = E_func(q, dq)


    # Controller torque
    C = controller(q, dq, t, desTraj, L1, L2, m1, m2, r1, r2, I1, I2, K, V, E_hat)

    #Calcul des matrices d'inertie et de coriolis à partir de q et dq
    I = inertia_matrix(q, L1, L2, m1, m2, r1, r2, I1, I2)
    G = coriolis_matrix(q, dq, L1, L2, m2, r2)

    # Équation dynamique: I(q) ddq + G(q, dq) + E = C
    ddq = np.linalg.pinv(I)@(C-G)  #np.linalg.solve(I, C - G - E)
    
    dq = dq + ddq*t_max/100
    
    print('q', q)
    q = q + dq*t_max/100
    print('q t+1:', q  )
    
    print("q:", q, "dq:", dq, "ddq:", ddq)
        
    return q, dq, ddq

# --- Simulation ---
def simulate_movement(E_func=None, t_max=1.0):
    print("simulate_movement")
    """Simuler un mouvement avec ou sans champ de forces."""
    # Conditions initiales: q0 = [épaule, coude], dq0 = [0, 0]
    q0 = cond_init  # q1_0, q2_0, dq1_0, dq2_0, conditions initiales articulaires
    print(q0)
    t = np.linspace(0, t_max, 100) # 100 points de temps de simulation
    pos_list = []
    vel_list = []
    time, posX, velX, accelX , jerkX = fun_minjerktrajectory(0.5, x0, vx0, acx0, xf, vxf, acxf, 100)  # Pré-calculer la trajectoire désirée
    time, posY, velY, accelY , jerkY = fun_minjerktrajectory(0.5, y0, vy0, acy0, yf, vyf, acyf, 100)  # Pré-calculer la trajectoire désirée

    #Calcul des positions de la main au cours du temps (la trajectoire part de la position initiale de la main)
    trajX = handPos[0] + posX
    trajY = handPos[1] + posY   
    
    # Création d'un datafrale pour stocker la trajectoire désirée
    desTraj["time"] = time
    desTraj["posX"] = posX
    desTraj["posY"] = posY
    desTraj["velX"] = velX
    desTraj["velY"] = velY
    desTraj["accelX"] = accelX
    desTraj["accelY"] = accelY
    desTraj["jerkX"] = jerkX
    desTraj["jerkY"] = jerkY
    desTraj["trajX"] = trajX
    desTraj["trajY"] = trajY
    
    pos_list = list(zip(trajX, trajY))
    vel_list = list(np.sqrt(velX**2 + velY**2))
    
    theta_list = []
    pos_verif_list=[]

    #Compute desired joint angle trajectories (for the desired hand trajectory)
    for i in range(len(time)):
        X = [trajX[i], trajY[i]]
        theta = inverse_kinematics(X[0], X[1], L1, L2)
         # # Convertir en coordonnées articulaires
        q_des, dq_des, ddq_des = compute_joint_desired_states([trajX[i], trajY[i]],
                                             [velX[i], velY[i]],
                                             [accelX[i], accelY[i]],
                                             L1, L2,
                                             jacobian_fn=jacobian,
                                             jacobian_dot_fn=jacobian_dot,
                                             inverse_kin_fn=inverse_kinematics,
                                             use_damped_ls=True, damping_lambda=1e-3)
        theta_list.append((theta, dq_des, ddq_des))
        
        pos_verif = direct_kinematics(theta, L1, L2)
        pos_verif_list.append(pos_verif)

    plt.plot(*zip(*pos_verif_list), label="Trajectoire vérifiée", linestyle=':')
    plt.show()
    

    #Store desired joint angles in the dataframe
    desTraj["q1_des"] = [theta[0][0] for theta in theta_list]
    desTraj["q2_des"] = [theta[0][1] for theta in theta_list]
    desTraj["dq1_des"] = [theta[1][0] for theta in theta_list]
    desTraj["dq2_des"] = [theta[1][1] for theta in theta_list]
    desTraj["ddq1_des"] = [theta[2][0] for theta in theta_list]
    desTraj["ddq2_des"] = [theta[2][1] for theta in theta_list]

    # --- Visualisation des résultats de la trajectoire désirée ---
    fig, axs = plt.subplots(4, 1, figsize=(8, 8))
    # Trajectoire désirée (minimum jerk) en X et Y
    axs[0].plot(*zip(*pos_list), label="Trajectoire désirée", linestyle='--')
    # Marquer la position de départ (en vert) et d'arrivée (en rouge)
    axs[0].scatter(pos_list[0][0], pos_list[0][1], c='green', label="Départ")
    axs[0].scatter(pos_list[-1][0], pos_list[-1][1], c='red', label="Arrivée")
    axs[0].set_title("Trajectoire désirée (minimum jerk)")
    axs[0].set_xlabel("X (m)")
    axs[0].set_ylabel("Y (m)")
    axs[0].axis("equal")
    axs[0].legend()
    axs[0].grid()

    # Vitesse désirée (minimum jerk) en fonction du temps 
    axs[1].plot(time, vel_list, label="Vitesse désirée", linestyle='--')
    axs[1].set_title("Vitesse désirée (minimum jerk)")
    axs[1].set_xlabel("Temps (s)")
    axs[1].set_ylabel("Vitesse (m/s)")
    axs[1].legend()
    axs[1].grid()
    
    # Angles articulaires désirés en fonction du temps
    axs[2].plot(time, np.rad2deg(desTraj['q1_des']), label="q1 (rad)")
    axs[2].plot(time, np.rad2deg(desTraj['q2_des']), label="q2 (rad)")
    axs[2].set_title("Angles articulaires désirés")
    axs[2].set_xlabel("Temps (s)")
    axs[2].set_ylabel("Angle (rad)")
    axs[2].legend()
    axs[2].grid()
    
    # Vitesses articulaires désirées en fonction du temps
    axs[3].plot(time, np.rad2deg(desTraj['dq1_des']), label="dq1 (rad/s)")
    axs[3].plot(time, np.rad2deg(desTraj['dq2_des']), label="dq2 (rad/s)")
    axs[3].set_title("Vitesses articulaires désirées")
    axs[3].set_xlabel("Temps (s)")
    axs[3].set_ylabel("Vitesse (rad/s)")
    axs[3].legend()
    axs[3].grid()

    plt.tight_layout()
    plt.show()

    initial_conditions = cond_init
    
    print(cond_init)
    q_res, dq_res, ddq_res = [(cond_init[0], cond_init[1])], [(cond_init[2], cond_init[3])], [(0,0)]
    print("Time : ", time)
    for t in range(len(time)-1):
        q, dq, ddq=dynamics(initial_conditions, time[t], desTraj, K, V, L1, L2, m1, m2, r1, r2, I1, I2, E_func, E_hat=None)
        initial_conditions = np.concatenate([q, dq])
        q_res.append(q)
        dq_res.append(dq)
        ddq_res.append(ddq)
        # input("Press Enter to continue to the next step...")
    
    # sol = odeint(dynamics, initial_conditions, t,
    #              args=(desTraj, K, V, L1, L2, m1, m2, r1, r2, I1, I2, E_func, None))
    #q, dq = sol[:, :2], sol[:, 2:]
    
    # #new figure with q_res, ddq_res
    # fig, axs = plt.subplots(3, 1, figsize=(8, 8))
    # # Angles articulaires en fonction du temps
    # axs[0].plot(time, np.rad2deg([q[0] for q in q_res]), label="q1 (rad)")
    # axs[0].plot(time, np.rad2deg([q[1] for q in q_res]), label="q2 (rad)")
    # axs[0].set_title("Angles articulaires")
    return time, q_res, dq_res, ddq_res

# --- Visualisation ---
def plot_trajectory(t, q):
    print("plot_trajectory")
    """Tracer la trajectoire du point final (main)."""
    x, y = [], []
    for qi in q:
        xi = L1*np.cos(qi[0]) + L2*np.cos(qi[0]+qi[1])
        yi = L1*np.sin(qi[0]) + L2*np.sin(qi[0]+qi[1])
        x.append(xi)
        y.append(yi)
    plt.plot(x, y, label="Trajectoire")
    plt.scatter(x[0], y[0], c='green', label="Début")
    plt.scatter(x[-1], y[-1], c='red', label="Fin")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis("equal")
    plt.legend()
    plt.title("Trajectoire de la main")
    plt.grid()
    plt.show()
 

# --- Exemple d'utilisation ---
if __name__ == "__main__":
    # Simuler sans champ de forces
    t, q_null, _, _ = simulate_movement(E_func=None, t_max=t_max)
    print(q_null)
    plot_trajectory(t, q_null)

    # # Simuler avec champ de forces cartésien
    # t, q_field, _ = simulate_movement(E_func=endpoint_force_field)
    # plot_trajectory(t, q_field)
