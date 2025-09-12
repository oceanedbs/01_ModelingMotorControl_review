import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# --- Paramètres mécaniques du bras (Table 1 de l'article) ---
# Longueurs (m), masses (kg), inerties (kg·m²), centres de masse (m)

L1 = 0.33  # Longueur bras
L2 = 0.34  # Longueur avant-bras
m1 = 1.93  # Masse bras (parametres anthropometriques)
m2 = 1.52  # Masse avant-bras
I1 = 0.0141  # Inertie bras
I2 = 0.0188  # Inertie avant-bras
r1 = 0.165  # Centre de masse bras
r2 = 0.19   # Centre de masse avant-bras

# Matrices de stiffness et viscosité (N·m/rad et N·m·s/rad)
K = np.array([[-15, -6], [-6, -16]])  # Stiffness
V = np.array([[-2.3, -0.9], [-0.9, -2.4]])  # Viscosité

# --- Dynamique du bras ---
def inertia_matrix(q):
    print("inertia_matrix")
    """Matrice d'inertie I(q) pour un bras à 2 DDL."""
    q1, q2 = q
    I11 = I1 + I2 + m1*r1**2 + m2*(L1**2 + r2**2 + 2*L1*r2*np.cos(q2))
    I12 = I2 + m2*(r2**2 + L1*r2*np.cos(q2))
    I21 = I2 + m2*(r2**2 + L1*r2*np.cos(q2))
    I22 = I2 + m2*r2**2
    return np.array([[I11, I12], [I21, I22]])

def coriolis_matrix(q, dq):
    print("coriolis_matrix")
    """Forces de Coriolis et centrifuges G(q, dq)."""
    q1, q2 = q
    dq1, dq2 = dq
    G1 = -m2 * L1 * r2 * np.sin(q2) * dq2**2 - 2 * m2 * L1 * r2 * np.sin(q2) * dq1 * dq2
    G2 = m2 * L1 * r2 * np.sin(q2) * dq1**2
    return np.array([G1, G2])

# --- Champ de forces ---
def endpoint_force_field(x_dot):
    print("endpoint_force_field")
    """Champ de forces translation-invariant en coordonnées cartésiennes (Eq. 1)."""
    B = np.array([[-10.1, -11.2], [-11.2, 11.1]])
    return B @ x_dot

def joint_torque_field(q, dq):
    print("joint_torque_field")
    """Champ de forces translation-invariant en coordonnées articulaires (Eq. 3)."""
    # Jacobien simplifié (à calculer proprement pour une implémentation complète)
    J = np.array([[ -L1*np.sin(q[0]) - L2*np.sin(q[0]+q[1]), -L2*np.sin(q[0]+q[1]) ],
                  [ L1*np.cos(q[0]) + L2*np.cos(q[0]+q[1]),  L2*np.cos(q[0]+q[1]) ]])
    B = np.array([[-10.1, -11.2], [-11.2, 11.1]])
    W = J.T @ B @ J  # Eq. 4: W = J^T B J
    return W @ dq

# --- Trajectoire désirée (minimum jerk) ---
def desired_trajectory(t, t_total=0.65):
    print("desired_trajectory")
    """Trajectoire en ligne droite 'minimum jerk' pour le point final (x, y)."""
    # Paramètres pour un mouvement de 10 cm à 0° (à adapter)
    x0, y0 = 0, 0  # Position initiale
    xf, yf = 0.1, 0  # Position finale (10 cm à 0°)
    # Polynôme minimum jerk (Flash & Hogan, 1985)
    if t <= t_total:
        s = t / t_total
        x = x0 + (xf - x0) * (10*s**3 - 15*s**4 + 6*s**5)
        y = y0 + (yf - y0) * (10*s**3 - 15*s**4 + 6*s**5)
        dx = (xf - x0) * (30*s**2 - 60*s**3 + 30*s**4) / t_total
        dy = (yf - y0) * (30*s**2 - 60*s**3 + 30*s**4) / t_total
        ddx = (xf - x0) * (60*s - 180*s**2 + 120*s**3) / t_total**2
        ddy = (yf - y0) * (60*s - 180*s**2 + 120*s**3) / t_total**2
    else:
        x, y = xf, yf
        dx, dy = 0, 0
        ddx, ddy = 0, 0
    return np.array([x, y]), np.array([dx, dy]), np.array([ddx, ddy])

# --- Jacobien ---
def jacobian(q):
    print("jacobian")
    """Jacobien pour convertir vitesses articulaires en vitesses cartésiennes."""
    q1, q2 = q
    J = np.array([
        [-L1*np.sin(q1) - L2*np.sin(q1+q2), -L2*np.sin(q1+q2)],
        [ L1*np.cos(q1) + L2*np.cos(q1+q2),  L2*np.cos(q1+q2)]
    ])
    return J

def jacobian_dot(q, qd):
    print("jacobian_dot")
    q1, q2 = q
    dq1, dq2 = qd
    """Dérivée du jacobien (dJ/dt)."""
    return np.array([
        [-L1*np.cos(q1)*dq1 - L2*np.cos(q1+q2)*(dq1+dq2), -L2*np.cos(q1+q2)*(dq1+dq2)],
        [-L1*np.sin(q1)*dq1 - L2*np.sin(q1+q2)*(dq1+dq2), -L2*np.sin(q1+q2)*(dq1+dq2)]
    ])



def forward_kinematics_acceleration(q1, q2, dq1, dq2, ddq1, ddq2):
    print("forward_kinematics_acceleration")
    """Calcule les accélérations cartésiennes (ddx, ddy)."""
    J = jacobian(q1, q2)
    dJ = jacobian_dot(q1, q2, dq1, dq2)
    ddq = np.array([ddq1, ddq2])
    dq = np.array([dq1, dq2])
    ddx = J[0, 0]*ddq1 + J[0, 1]*ddq2 + dJ[0, 0]*dq1 + dJ[0, 1]*dq2
    ddy = J[1, 0]*ddq1 + J[1, 1]*ddq2 + dJ[1, 0]*dq1 + dJ[1, 1]*dq2
    return ddx, ddy

def inverse_jacobian(q1, q2):
    print("inverse_jacobian")
    """Inverse le jacobien (si possible)."""
    J = jacobian([q1, q2])
    det_J = np.linalg.det(J)
    if abs(det_J) < 1e-6:
        raise ValueError("Jacobien singulier (det ≈ 0)")
    return np.linalg.inv(J)

def direct_kinematics(q):
    print("direct_kinematics")
    """Calcule la position (x, y) du point final à partir de q."""
    q1, q2 = q
    x = L1 * np.cos(q1) + L2 * np.cos(q1 + q2)
    y = L1 * np.sin(q1) + L2 * np.sin(q1 + q2)
    return np.array([x, y])

def inverse_kinematics(x, y):
    print("inverse_kinematics")
    """Calcule q1 et q2 pour atteindre (x, y)."""
    # Distance à la cible
    d = np.sqrt(x**2 + y**2)
    if d > L1 + L2:
        raise ValueError("Cible hors de portée (d > L1 + L2)")

    # Angle de l'épaule (approximation initiale)
    q1_guess = np.arctan2(y, x)

    # Calcul de q2 (coude)
    c2 = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    if abs(c2) > 1:  # Problème de précision numérique
        c2 = np.clip(c2, -1, 1)
    q2 = np.arccos(c2)  # Solution "coude bas" (utiliser -arccos pour "coude haut")

    # Calcul de q1 (épaule)
    k1 = L1 + L2 * np.cos(q2)
    k2 = L2 * np.sin(q2)
    q1 = np.arctan2(y, x) - np.arctan2(k2, k1)

    return np.array([q1, q2])

def inverse_kinematics_velocities(x, dx):
    print("inverse_kinematics_velocities")
    """Calcule q1, q2 et leurs vitesses à partir de x, y, dx, dy."""
    x, y = x
    dx, dy = dx
    
    # Cinématique inverse pour q1, q2
    q1, q2 = inverse_kinematics(x, y)

    # Jacobien et son inverse
    J = jacobian((q1, q2))
    try:
        J_inv = np.linalg.inv(J)
    except np.linalg.LinAlgError:
        # Si le jacobien est singulier, utiliser le pseudo-inverse
        J_inv = np.linalg.pinv(J)

    # Vitesses articulaires
    dq = J_inv @ np.array([dx, dy])
    return q1, q2, dq[0], dq[1]

def inverse_kinematics_accelerations(x, dx, ddx):
    print("inverse_kinematics_accelerations")
    """Calcule q1, q2, dq1, dq2, ddq1, ddq2."""
    x, y = x
    dx, dy = dx
    ddx, ddy = ddx
    q1, q2, dq1, dq2 = inverse_kinematics_velocities((x, y), (dx, dy))
    J = jacobian((q1, q2))
    dJ = jacobian_dot((q1, q2), (dq1, dq2))
    J_inv = np.linalg.inv(J)
    ddq = J_inv @ (np.array([ddx, ddy]) - dJ @ np.array([dq1, dq2]))
    return (q1, q2), (dq1, dq2), (ddq[0], ddq[1])

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
    y0 = np.array([np.deg2rad(45), np.deg2rad(90), 0, 0])  # q1_0, q2_0, dq1_0, dq2_0, conditions initiales articulaires
    t = np.linspace(0, t_max, 100) # 100 points de temps de simulation
    pos_list = []
    vel_list = []
    position_list = []
    for i in range(len(t)-1):
        pos, vel, _ = desired_trajectory(t[i])  # Pré-calculer la trajectoire désirée
        pos_list.append(pos)
        vel = np.sqrt(vel[0]**2 + vel[1]**2)
        vel_list.append(vel)
        x, y = inverse_kinematics(pos[0], pos[1])
        position_list.append((x, y))


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
    axs[1].plot(t[1:], vel_list, label="Vitesse désirée", linestyle='--')
    axs[1].set_title("Vitesse désirée (minimum jerk)")
    axs[1].set_xlabel("Temps (s)")
    axs[1].set_ylabel("Vitesse (m/s)")
    axs[1].legend()
    axs[1].grid()
    
    # Tracer les deux colonnes de position_list (q1 et q2) en fonction du temps
    position_array = np.array(position_list)
    axs[2].plot(t[1:], np.rad2deg(position_array[:, 0]), label="q1 (rad)")
    axs[2].plot(t[1:], np.rad2deg(position_array[:, 1]), label="q2 (rad)")
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
