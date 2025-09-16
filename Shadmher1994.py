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

handPos = direct_kinematics([np.deg2rad(45), np.deg2rad(90)], L1, L2)

# --- Contrôleur (Eq. 9) ---
def controller(q, dq, t, E_hat=None):
    print("controller")
    """Contrôleur avec modèle interne et feedback viscoélastique."""
    
    #Retrieve desired trajectory at time t
   
    if t > desTraj["time"].iloc[-1]:
        t = desTraj["time"].iloc[-1]
    x_des = desTraj["trajX"].values[np.searchsorted(desTraj["time"].values, t)]
    y_des = desTraj["trajY"].values[np.searchsorted(desTraj["time"].values, t)]
    x_dot_des = desTraj["velX"].values[np.searchsorted(desTraj["time"].values, t)]
    y_dot_des = desTraj["velY"].values[np.searchsorted(desTraj["time"].values, t)]
    x_dd_des = desTraj["accelX"].values[np.searchsorted(desTraj["time"].values, t)]
    y_dd_des = desTraj["accelY"].values[np.searchsorted(desTraj["time"].values, t)]
    x_des = np.array([x_des, y_des])
    x_dot_des = np.array([x_dot_des, y_dot_des])
    x_dd_des = np.array([x_dd_des, y_dd_des])
    
    
    # Convertir en coordonnées articulaires
    q_des, dq_des, ddq_des = inverse_kinematics_accelerations(x_des, x_dot_des, x_dd_des, L1, L2) #position articulaire désirée via la cinématique inverse

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
    C = D_hat + E_hat - S
    return C

# --- Dynamique complète (Eq. 4) ---
def dynamics(y, t, E_func=None):
    print("dynamics")
    """Équations différentielles pour la dynamique du bras."""
    q, dq = y[:2], y[2:] #récupérer q et dq initiaux
 
 
    #Calcul des matrices d'inertie et de coriolis à partir de q et dq
    I = inertia_matrix(q, L1, L2, m1, m2, r1, r2, I1, I2)
    G = coriolis_matrix(q, dq, L1, L2, m2, r2)

    # Champ de forces externe
    if E_func is not None:
        if E_func.__name__ == "endpoint_force_field":
            J = ddirect_kinematics(q, L1, L2)
            x_dot = J @ dq
            f = E_func(x_dot)
            E = J.T @ f  # Convertir en torques articulaires
        else:
            E = E_func(q, dq)
    else:
        E = np.zeros(2)

    # Contrôleur (avec E_hat = None initialement)
    C = controller(q, dq, t, E_hat=None)

    # Équation dynamique: I(q) ddq + G(q, dq) + E = C
    ddq = np.linalg.inv(I) @ (C - G - E)
    
    return np.concatenate([dq, ddq])

# --- Simulation ---
def simulate_movement(E_func=None, t_max=1.0):
    print("simulate_movement")
    """Simuler un mouvement avec ou sans champ de forces."""
    # Conditions initiales: q0 = [épaule, coude], dq0 = [0, 0]
    q0 = cond_init  # q1_0, q2_0, dq1_0, dq2_0, conditions initiales articulaires
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

    #Compute desired joint angle trajectories (for the desired hand trajectory)
    for i in range(len(time)):
        X = [trajX[i], trajY[i]]
        theta = inverse_kinematics(X[0], X[1], L1, L2)
        theta_list.append(theta)

    #Store desired joint angles in the dataframe
    desTraj["q1_des"] = [theta[0] for theta in theta_list]
    desTraj["q2_des"] = [theta[1] for theta in theta_list]


    # --- Visualisation des résultats de la trajectoire désirée ---
    fig, axs = plt.subplots(3, 1, figsize=(8, 8))
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

    plt.tight_layout()
    plt.show()
    
    initial_conditions = cond_init
    sol = odeint(dynamics, initial_conditions, time, args=(E_func,))
    q, dq = sol[:, :2], sol[:, 2:]
    return t, q, dq

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
    t, q_null, _ = simulate_movement(E_func=None, t_max=0.5)
    plot_trajectory(t, q_null)

    # # Simuler avec champ de forces cartésien
    # t, q_field, _ = simulate_movement(E_func=endpoint_force_field)
    # plot_trajectory(t, q_field)
