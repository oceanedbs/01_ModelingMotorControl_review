import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches    

# --- Dynamique du bras ---
def inertia_matrix(q, L1, L2, m1, m2, r1, r2, I1, I2):
    """Matrice d'inertie I(q) pour un bras à 2 DDL."""
    q1, q2 = q
    I11 = I1 + I2 + m1*r1**2 + m2*(L1**2 + r2**2 + 2*L1*r2*np.cos(q2))
    I12 = I2 + m2*(r2**2 + L1*r2*np.cos(q2))
    I21 = I2 + m2*(r2**2 + L1*r2*np.cos(q2))
    I22 = I2 + m2*r2**2
    return np.array([[I11, I12], [I21, I22]])

def coriolis_matrix(q, dq, L1, L2, m2, r2):
    """Forces de Coriolis et centrifuges G(q, dq)."""
    q1, q2 = q
    dq1, dq2 = dq
    h = m2 * L1 * r2 * np.sin(q2)
    G1 = -h * (2 * dq1 * dq2 + dq2**2)
    G2 =  h * (dq1**2)
    # G1 = -m2 * L1 * r2 * np.sin(q2) * dq2**2 - 2 * m2 * L1 * r2 * np.sin(q2) * dq1 * dq2
    # G2 = m2 * L1 * r2 * np.sin(q2) * dq1**2
    return np.array([G1, G2])

# --- Champ de forces ---
def endpoint_force_field(x_dot):
    """Champ de forces translation-invariant en coordonnées cartésiennes (Eq. 1)."""
    B = np.array([[-10.1, -11.2], [-11.2, 11.1]])
    return B @ x_dot

def joint_torque_field(q, dq, L1, L2):
    """Champ de forces translation-invariant en coordonnées articulaires (Eq. 3)."""
    # Jacobien simplifié (à calculer proprement pour une implémentation complète)
    J = np.array([[ -L1*np.sin(q[0]) - L2*np.sin(q[0]+q[1]), -L2*np.sin(q[0]+q[1]) ],
                  [ L1*np.cos(q[0]) + L2*np.cos(q[0]+q[1]),  L2*np.cos(q[0]+q[1]) ]])
    B = np.array([[-10.1, -11.2], [-11.2, 11.1]])
    W = J.T @ B @ J  # Eq. 4: W = J^T B J*
    return W @ dq


def fun_minjerkcoeffs(duration, posi, veli, acci, posf, velf, accf):
    """
    Calcule les coefficients du polynôme de minimum jerk reliant les conditions initiales et finales.

    Parameters
    ----------
    duration : float
        Durée totale du mouvement.
    posi : float
        Position initiale.
    veli : float
        Vitesse initiale.
    acci : float
        Accélération initiale.
    posf : float
        Position finale.
    velf : float
        Vitesse finale.
    accf : float
        Accélération finale.

    Returns
    -------
    coeffs : list of float
        Coefficients [b0, b1, b2, b3, b4, b5] du polynôme de degré 5.
    """
    T = duration
    b0 = posi
    b1 = veli
    b2 = acci / 2
    A = (posf - b0 - b1 * T - b2 * T * T) / (T * T * T)
    B = (velf - b1 - 2 * b2 * T) / (T * T)
    C = (accf - 2 * b2) / T
    b5 = (C - 6 * B + 12 * A) / (2 * T * T)
    b4 = (B - 3 * A) / T - 2 * T * b5
    b3 = A - T * b4 - T * T * b5

    return [b0, b1, b2, b3, b4, b5]


def fun_minjerktrajectory(duration, posi, veli, acci, posf, velf, accf, npoints):
    """
    Generates a minimum-jerk trajectory given initial and final boundary conditions.

    This function computes the position, velocity, acceleration, and jerk profiles for a movement
    that minimizes the jerk (third derivative of position) over a specified duration, using the
    provided initial and final positions, velocities, and accelerations.

    Parameters
    ----------
    duration : float
        Total duration of the trajectory (in seconds).
    posi : float
        Initial position.
    veli : float
        Initial velocity.
    acci : float
        Initial acceleration.
    posf : float
        Final position.
    velf : float
        Final velocity.
    accf : float
        Final acceleration.
    npoints : int
        Number of points to sample along the trajectory.

    Returns
    -------
    mytime : numpy.ndarray
        Array of time points at which the trajectory is sampled.
    position : numpy.ndarray
        Array of positions at each time point.
    velocity : numpy.ndarray
        Array of velocities at each time point.
    acceleration : numpy.ndarray
        Array of accelerations at each time point.
    jerk : numpy.ndarray
        Array of jerks at each time point.

    Notes
    -----
    This function relies on `fun_minjerkcoeffs` to compute the polynomial coefficients for the
    minimum-jerk trajectory.
    """
    coeff = fun_minjerkcoeffs(duration, posi, veli, acci, posf, velf, accf)

    mytime = np.linspace(0, duration, npoints)

    c1 = coeff[0]
    c2 = coeff[1]
    c3 = coeff[2]
    c4 = coeff[3]
    c5 = coeff[4]
    c6 = coeff[5]
    position = np.zeros(npoints)
    velocity = np.zeros(npoints)
    acceleration = np.zeros(npoints)
    jerk = np.zeros(npoints)
    for n in range(0, npoints):
        t1 = mytime[n]
        t2 = t1*t1
        t3 = t1*t2
        t4 = t1*t3
        t5 = t1*t4
        position[n] = c1 + c2*t1 + c3*t2 + c4*t3 + c5*t4 + c6*t5
        velocity[n] = c2 + 2*c3*t1 + 3*c4*t2 + 4*c5*t3 + 5*c6*t4
        acceleration[n] = 2*c3 + 6*c4*t1 + 12*c5*t2 + 20*c6*t3
        jerk[n] = 6*c4 + 24*c5*t1 + 60*c6*t2
    
    return mytime, position, velocity, acceleration, jerk


def direct_kinematics(q, L1, L2):
    """Calcule la position (x, y) du point final à partir de q."""
    q1, q2 = q
    x = L1 * np.cos(q1) + L2 * np.cos(q1 + q2)
    y = L1 * np.sin(q1) + L2 * np.sin(q1 + q2)
    return np.array([x, y])

def inverse_kinematics(x, y, L1, L2):
    """Calcule q1 et q2 pour atteindre (x, y)."""
    # Distance à la cible
    d2 = (x**2 + y**2)
    u = (d2 - L1**2 - L2**2)/(2*L1*L2)
    u = max(min(u, 1.0), -1.0)
        
    q1 = math.acos(u)
    q2 = math.atan2(y, x) - math.atan2(L2*np.sin(q1), L1+L2*np.cos(q1))
    
    if q1 < 0:
        q1 = q1 + 2*math.pi

    return np.array([q2, q1])


# --- Jacobien ---
def jacobian(q, L1, L2):
    """Dérivé du modèle cinématique direct pour convertire les vitesses articulaires en vitesses cartésiennes."""
    q1, q2 = q
    J = np.array([
        [-L1*np.sin(q1) - L2*np.sin(q1+q2), -L2*np.sin(q1+q2)],
        [ L1*np.cos(q1) + L2*np.cos(q1+q2),  L2*np.cos(q1+q2)]
    ])
    return J

#%%%%%%%

def jacobian_dot(q, qd, L1, L2):
    q1, q2 = q
    dq1, dq2 = qd
    """Dérivée du jacobien (dJ/dt)."""
    return np.array([
        [-L1*np.cos(q1)*dq1 - L2*np.cos(q1+q2)*(dq1+dq2), -L2*np.cos(q1+q2)*(dq1+dq2)],
        [-L1*np.sin(q1)*dq1 - L2*np.sin(q1+q2)*(dq1+dq2), -L2*np.sin(q1+q2)*(dq1+dq2)]
    ])



def forward_kinematics_acceleration(q1, q2, dq1, dq2, ddq1, ddq2):
    """Calcule les accélérations cartésiennes (ddx, ddy)."""
    J = jacobian(q1, q2)
    dJ = jacobian_dot(q1, q2, dq1, dq2)
    ddq = np.array([ddq1, ddq2])
    dq = np.array([dq1, dq2])
    ddx = J[0, 0]*ddq1 + J[0, 1]*ddq2 + dJ[0, 0]*dq1 + dJ[0, 1]*dq2
    ddy = J[1, 0]*ddq1 + J[1, 1]*ddq2 + dJ[1, 0]*dq1 + dJ[1, 1]*dq2
    return ddx, ddy

def inverse_jacobian(q1, q2):
    """Inverse le jacobien (si possible)."""
    J = jacobian([q1, q2], L1, L2)
    det_J = np.linalg.det(J)
    if abs(det_J) < 1e-6:
        raise ValueError("Jacobien singulier (det ≈ 0)")
    return np.linalg.inv(J)

def inverse_kinematics_velocities(x, dx, L1, L2):
    """Calcule q1, q2 et leurs vitesses à partir de x, y, dx, dy."""
    x, y = x
    dx, dy = dx
    
    # Cinématique inverse pour q1, q2
    q1, q2 = inverse_kinematics(x, y, L1, L2)

    # Jacobien et son inverse
    J = jacobian((q1, q2), L1, L2)
    try:
        J_inv = np.linalg.inv(J)
    except np.linalg.LinAlgError:
        # Si le jacobien est singulier, utiliser le pseudo-inverse
        J_inv = np.linalg.pinv(J)

    # Vitesses articulaires
    dq = J_inv @ np.array([dx, dy])
    return q1, q2, dq[0], dq[1]

def compute_joint_desired_states(x_des, dx_des, ddx_des, L1, L2,
                                jacobian_fn, jacobian_dot_fn,
                                inverse_kin_fn,
                                use_damped_ls=False, damping_lambda=1e-3):
    """
    Compute qd, dqd, ddqd from desired Cartesian x_des, dx_des, ddx_des.
    Inputs:
      x_des : array-like (2,) desired [x, y]
      dx_des: array-like (2,) desired [xdot, ydot]
      ddx_des: array-like (2,) desired [xddot, yddot]
      L1,L2 : link lengths
      jacobian_fn(q, L1, L2) -> 2x2 Jacobian
      jacobian_dot_fn(q, qd, L1, L2) -> 2x2 dJ/dt
      inverse_kin_fn(x, y, L1, L2) -> [q1,q2]
      use_damped_ls : if True, use damped least squares for Jacobian inversion
      damping_lambda : damping factor for DLS

    Returns:
      qd : np.array (2,)
      dqd: np.array (2,)
      ddqd: np.array (2,)
    """

    x_des = np.asarray(x_des, dtype=float)
    
    dx_des = np.asarray(dx_des, dtype=float)
    ddx_des = np.asarray(ddx_des, dtype=float)

    # 1) inverse kinematics (position)
    qd = np.asarray(inverse_kin_fn(x_des[0], x_des[1], L1, L2), dtype=float)  # shape (2,)

    # 2) Jacobian at qd
    J = np.asarray(jacobian_fn(qd, L1, L2), dtype=float)  # shape (2,2)

    # invert J robustly
    def invert_J(J):
        # try direct inverse if well-conditioned
        try:
            det = np.linalg.det(J)
        except Exception:
            det = 0.0
        if abs(det) > 1e-8:
            return np.linalg.inv(J)
        # fallback: pseudoinverse
        if not use_damped_ls:
            return np.linalg.pinv(J)
        # damped least squares inverse: J^T (J J^T + lambda^2 I)^{-1}
        JT = J.T
        JJt = J @ JT
        reg = JJt + (damping_lambda**2) * np.eye(JJt.shape[0])
        inv_reg = np.linalg.inv(reg)
        return JT @ inv_reg

    J_inv = invert_J(J)

    # 3) joint velocities
    dqd = J_inv @ dx_des  # shape (2,)

    # 4) jacobian time derivative at (q, dqd)
    dJ = np.asarray(jacobian_dot_fn(qd, dqd, L1, L2), dtype=float)

    # 5) joint accelerations: ddq = J^{-1} (ddx - dJ * dq)
    rhs = ddx_des - dJ @ dqd
    ddqd = J_inv @ rhs

    return qd, dqd, ddqd


def plot_arm(init_config, L1, L2, target):
    fig, axs = plt.subplots(1, 1, layout="constrained")

    targetx = target[0]
    targety = target[1]
    shpos = [0, 0]
    elpos = direct_kinematics([init_config[0], init_config[1]], L1, 0)
    handpos = direct_kinematics([init_config[0], init_config[1]], L1, L2)
    xlim, ylim, xticks, yticks = fun_axisfunscaling(-.27+.1, .27+.1, -0.02, 0.52, 10, True)
    axs.plot([shpos[0], elpos[0]], [shpos[1], elpos[1]], color='k', linewidth=3*0.5)
    axs.plot([elpos[0], handpos[0]], [elpos[1], handpos[1]], color='k', linewidth=3*0.5)
    circle1 = plt.Circle((shpos[0], shpos[1]), 0.01, color='k', fill=True, linewidth=0.5)
    circle2 = plt.Circle((elpos[0], elpos[1]), 0.01, color='k', fill=True, linewidth=0.5)
    axs.add_patch(circle1)
    axs.add_patch(circle2)
    elpos2 = direct_kinematics([init_config[0], init_config[1]], 1.5*L1, 0)
    axs.plot([elpos[0], elpos2[0]], [elpos[1], elpos2[1]], color='k', linewidth=0.5*0.5)
    axs.set(xlim=xlim, ylim=ylim)
    axs.set_axis_off()

    axs.plot([0, .2], [0, 0], color='k', linewidth=0.5*0.5)
    axs.text(0.21, -0.01, '$x$')
    axs.plot([0, 0], [0, .2], color='k', linewidth=0.5*0.5)
    axs.text(-0.01, 0.21, '$y$')

    axs.plot([handpos[0], handpos[0]+.1], [handpos[1], handpos[1]], color='k', linewidth=0.5*0.5)
    axs.text(handpos[0]+.11, handpos[1]-0.01, '0$^\\circ$')

    axs.text(0.08, 0.025, '$\\theta_\\mathrm{sh}$')
    e = patches.Arc((0, 0), .1, .1, angle=0.0, theta1=0.0, theta2=45.0)
    axs.add_patch(e)
    axs.text(0.19, 0.29, '$\\theta_\\mathrm{el}$')
    e = patches.Arc((elpos[0], elpos[1]), .1, .1, angle=0.0, theta1=45, theta2=135.0)
    axs.add_patch(e)
    axs.text(-0.12, 0.44, '($x$,$y$)')

    e = patches.Arc((0.1, 0.1), .1, .1, angle=0.0, theta1=0.0, theta2=85.0)
    axs.add_patch(e)
    axs.annotate("", xytext=(0.12, 0.146), xy=(0.095, 0.154),
                arrowprops=dict(color='k', arrowstyle="->"))
    axs.text(0.09, 0.17, '$\\tau_\\mathrm{sh}$')

    e = patches.Arc((0.1, 0.31), .1, .1, angle=0.0, theta1=65, theta2=165.0)
    axs.add_patch(e)
    axs.annotate("", xytext=(0.055, 0.333), xy=(0.047, 0.315),
                arrowprops=dict(color='k', arrowstyle="->"))
    axs.text(0.04, 0.285, '$\\tau_\\mathrm{el}$')

    circle1 = plt.Circle((handpos[0], handpos[1]), 0.004, color='k', fill=False, linewidth=0.5)
    circle2 = plt.Circle((handpos[0]+targetx, handpos[1]+targety), 0.008,
                        color='k', fill=False, linewidth=0.5)
    axs.add_patch(circle1)
    axs.add_patch(circle2)

    axs.plot([-0.05, -0.05], [0, 0.05], color='k', linewidth=0.5*0.5)



def fun_axisfunscaling(xmin, xmax, ymin, ymax, axisscaling, zeroref):
    xmin1 = xmin
    xmax1 = xmax
    ymin1 = ymin
    ymax1 = ymax
    if xmin >= 0:
        if zeroref:
            xmin1 = 0
        else:
            xmin1 = xmin*(1-axisscaling/100.0)
        xmax1 = xmax*(1+axisscaling/100.0)

    if xmax <= 0:
        if zeroref:
            xmax1 = 0
        else:
            xmax1 = xmax*(1-axisscaling/100.0)
        xmin1 = xmin*(1+axisscaling/100.0)

    if ymin >= 0:
        if zeroref:
            ymin1 = 0
        else:
            ymin1 = ymin*(1-axisscaling/100.)
        ymax1 = ymax*(1+axisscaling/100.0)

    if ymax <= 0:
        if zeroref:
            ymax1 = 0
        else:
            ymax1 = ymax*(1-axisscaling/100.0)
        ymin1 = ymin*(1+axisscaling/100.0)

    if (xmin < 0) & (xmax > 0):
        xmin1 = xmin*(1+axisscaling/100.0)
        xmax1 = xmax*(1+axisscaling/100.0)

    if (ymin < 0) & (ymax > 0):
        ymin1 = ymin*(1+axisscaling/100.0)
        ymax1 = ymax*(1+axisscaling/100.0)

    return fun_axisfun(xmin1, xmax1, ymin1, ymax1)



def fun_axisfun(xmin, xmax, ymin, ymax):
    if ((xmax == xmin) | (xmax < xmin)):
        xmin = 0
        xmax = 1

    if ((ymax == ymin) | (ymax < ymin)):
        ymin = 0
        ymax = 1

    myaxis = [xmin, xmax, ymin, ymax]
    xlim = [xmin, xmax]
    ylim = [ymin, ymax]

    tx = fun_ticksfun(xmin, xmax)
    ty = fun_ticksfun(ymin, ymax)
    return xlim, ylim, tx, ty



def fun_ticksfun(a, b):

    spacing = 0

    if ((a < 0) & (b > 0)):
        ro = max([b, -a])
        sc = 2
    else:
        ro = b - a
        sc = 1

    if ro == 0:
        print('error ticksfun')

    if ro < 0.001:
        spacing = 0.0002

    if ((ro >= 0.001) & (ro < 0.002)):
        spacing = 0.0005

    if ((ro >= 0.002) & (ro < 0.005)):
        spacing = 0.001

    if ((ro >= 0.005) & (ro < 0.01)):
        spacing = 0.002

    if ((ro >= 0.01) & (ro < 0.02)):
        spacing = 0.005

    if ((ro >= 0.02) & (ro < 0.05)):
        spacing = 0.01

    if ((ro >= 0.05) & (ro < 0.1)):
        spacing = 0.02

    if ((ro >= 0.1) & (ro < 0.2)):
        spacing = 0.05

    if ((ro >= 0.2) & (ro < 0.5)):
        spacing = 0.1

    if ((ro >= 0.5) & (ro < 1)):
        spacing = 0.2

    if ((ro >= 1) & (ro < 2)):
        spacing = 0.5

    if ((ro >= 2) & (ro < 5)):
        spacing = 1

    if ((ro >= 5) & (ro < 10)):
        spacing = 2

    if ((ro >= 10) & (ro < 20)):
        spacing = 5

    if ((ro >= 20) & (ro < 40)):
        spacing = 10

    if ((ro >= 40) & (ro <= 100)):
        spacing = 20

    if ((ro > 100) & (ro < 200)):
        spacing = 50

    if ((ro >= 200) & (ro < 500)):
        spacing = 100

    if ((ro >= 500) & (ro < 1000)):
        spacing = 250

    if ((ro >= 1000) & (ro < 2000)):
        spacing = 500

    if ((ro >= 2000) & (ro < 5000)):
        spacing = 1000

    if ((ro >= 5000) & (ro < 10000)):
        spacing = 2500

    spacing = spacing*sc

    if ((a < 0) & (b > 0)):
        N1 = int(-a/spacing)
        N2 = int(b/spacing)
        r1 = -spacing - np.arange(0, N1) * spacing
        r2 = 0 + np.arange(0, N2+1) * spacing
        r = np.concatenate((r1, r2), axis=0)
    else:
        if b == 0:
            N = int((b-a)/spacing)
            r = b - np.arange(0, N+1) * spacing
        else:
            r = np.linspace(a, b, 1 + int((b-a)/spacing))  # a:spacing:b
            N = int((b-a)/spacing)
            r = a + np.arange(0, N+1) * spacing

    return r
