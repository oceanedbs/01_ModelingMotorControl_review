import numpy as np
import math
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from TwoDLArm_func import *
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

# Matrices de stiffness et viscosité (N·m/rad et N·m·s/rad)
K = np.array([[-15, -6], [-6, -16]])  # Stiffness
V = np.array([[-2.3, -0.9], [-0.9, -2.4]])  # Viscosité

handPos = direct_kinematics([np.deg2rad(45), np.deg2rad(90)], L1, L2)


# # --- Trajectoire désirée (minimum jerk) ---
# def desired_trajectory(t, t_total=0.65):
#     print("desired_trajectory")
#     """Trajectoire en ligne droite 'minimum jerk' pour le point final (x, y)."""
#     # Paramètres pour un mouvement de 10 cm à 0° (à adapter)
#     x0, y0 = 0, 0  # Position initiale
#     xf, yf = 0.1, 0  # Position finale (10 cm à 0°)
#     # Polynôme minimum jerk (Flash & Hogan, 1985)
#     if t <= t_total:
#         s = t / t_total
#         x = x0 + (xf - x0) * (10*s**3 - 15*s**4 + 6*s**5)
#         y = y0 + (yf - y0) * (10*s**3 - 15*s**4 + 6*s**5)
#         dx = (xf - x0) * (30*s**2 - 60*s**3 + 30*s**4) / t_total
#         dy = (yf - y0) * (30*s**2 - 60*s**3 + 30*s**4) / t_total
#         ddx = (xf - x0) * (60*s - 180*s**2 + 120*s**3) / t_total**2
#         ddy = (yf - y0) * (60*s - 180*s**2 + 120*s**3) / t_total**2
#     else:
#         x, y = xf, yf
#         dx, dy = 0, 0
#         ddx, ddy = 0, 0
#     return np.array([x, y]), np.array([dx, dy]), np.array([ddx, ddy])


# --- Contrôleur (Eq. 9) ---
def controller(q, dq, t, E_hat=None):
    print("controller")
    """Contrôleur avec modèle interne et feedback viscoélastique."""
    # Trajectoire désirée en cartesien pour l'instant t
    x_des, x_dot_des, x_dd_des = desired_trajectory(t)
    
    # Convertir en coordonnées articulaires
    q_des, dq_des, ddq_des = inverse_kinematics_accelerations(x_des, x_dot_des, x_dd_des) #position articulaire désirée via la cinématique inverse

    # Modèle interne du bras (D_hat)
    I = inertia_matrix(q)
    G = coriolis_matrix(q, dq)
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
    print(q, dq)
    #Calcul des matrices d'inertie et de coriolis à partir de q et dq
    I = inertia_matrix(q)
    G = coriolis_matrix(q, dq)

    # Champ de forces externe
    if E_func is not None:
        if E_func.__name__ == "endpoint_force_field":
            J = jacobian(q)
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
    return np.concatenate([q, dq])

# --- Simulation ---
def simulate_movement(E_func=None, t_max=1.0):
    print("simulate_movement")
    """Simuler un mouvement avec ou sans champ de forces."""
    # Conditions initiales: q0 = [épaule, coude], dq0 = [0, 0]
    q0 = np.array([np.deg2rad(45), np.deg2rad(90), 0, 0])  # q1_0, q2_0, dq1_0, dq2_0, conditions initiales articulaires
    t = np.linspace(0, t_max, 100) # 100 points de temps de simulation
    pos_list = []
    vel_list = []
    time, posX, velX, accelX , _ = fun_minjerktrajectory(0.5, x0, vx0, acx0, xf, vxf, acxf, 100)  # Pré-calculer la trajectoire désirée
    time, posY, velY, accelY , _ = fun_minjerktrajectory(0.5, y0, vy0, acy0, yf, vyf, acyf, 100)  # Pré-calculer la trajectoire désirée
    
 
    
    trajX = handPos[0] + posX
    trajY = handPos[1] + posY   
    
    print(trajX)
    print(trajY)
    
    pos_list = list(zip(posX, posY))
    vel_list = list(np.sqrt(velX**2 + velY**2))
    
    theta_list = []

    for i in range(len(time)):
        X = [trajX[i], trajY[i]]
        print(X)
        theta = inverse_kinematics(X[0], X[1], L1, L2)
        theta_list.append(theta)
        print(theta)


    fig, axs = plt.subplots(3, 1, figsize=(8, 8))
    # Trajectoire désirée (minimum jerk)
    axs[0].plot(*zip(*pos_list), label="Trajectoire désirée", linestyle='--')
    axs[0].set_title("Trajectoire désirée (minimum jerk)")
    axs[0].set_xlabel("X (m)")
    axs[0].set_ylabel("Y (m)")
    axs[0].axis("equal")
    axs[0].legend()
    axs[0].grid()

    # Vitesse désirée (minimum jerk)
    axs[1].plot(time, vel_list, label="Vitesse désirée", linestyle='--')
    axs[1].set_title("Vitesse désirée (minimum jerk)")
    axs[1].set_xlabel("Temps (s)")
    axs[1].set_ylabel("Vitesse (m/s)")
    axs[1].legend()
    axs[1].grid()
    
    # Tracer les deux colonnes de position_list (q1 et q2) en fonction du temps
    theta_array = np.array(theta_list)
    # Extraire les colonnes de theta_array pour q1 et q2
    q1 = theta_array[:, 0]
    q2 = theta_array[:, 1]
    
    print(q1)
    axs[2].plot(time, np.rad2deg(q1), label="q1 (rad)")
    axs[2].plot(time, np.rad2deg(q2), label="q2 (rad)")
    axs[2].set_title("Angles articulaires désirés")
    axs[2].set_xlabel("Temps (s)")
    axs[2].set_ylabel("Angle (rad)")
    axs[2].legend()
    axs[2].grid()

    plt.tight_layout()
    plt.show()
    sol = odeint(dynamics, y0, t, args=(E_func,))
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
    t, q_null, _ = simulate_movement(E_func=None)
    plot_trajectory(t, q_null)

    # # Simuler avec champ de forces cartésien
    # t, q_field, _ = simulate_movement(E_func=endpoint_force_field)
    # plot_trajectory(t, q_field)
