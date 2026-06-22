"""
UR5 Kinematics – FK + IK
"""

import numpy as np

# PARÂMETROS DH

D1, D4, D5, D6 = 0.089159, 0.10915, 0.09465, 0.1788
A2, A3         = -0.425, -0.39225


DH = [
    [ np.pi/2,    0,   D1],   # DH[0] → junta 1
    [ 0,         A2,    0],   # DH[1] → junta 2
    [ 0,         A3,    0],   # DH[2] → junta 3
    [ np.pi/2,    0,   D4],   # DH[3] → junta 4
    [-np.pi/2,    0,   D5],   # DH[4] → junta 5
    [ 0,          0,   D6],   # DH[5] → junta 6
]

# Offset entre zero do CoppeliaSim (robô em pé) e zero do modelo DH (robô deitado)
JOINT_OFFSET = np.array([0, -90, 0, -90, 0, 0], dtype=float)


# CINEMÁTICA DIRETA (FK)

def dh_transform(alpha, a, d, theta):
    """
    Transformação DH padrão (Keating 2014):
    T = Rz(θ) · Tz(d) · Tx(a) · Rx(α)
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct,  -st*ca,  st*sa,  a*ct],
        [st,   ct*ca, -ct*sa,  a*st],
        [ 0,    sa,    ca,     d   ],
        [ 0,    0,     0,      1   ]
    ])


def fk(thetas_deg):
    thetas = np.radians(thetas_deg)
    T = np.eye(4)
    chain = [T.copy()]
    for (alpha, a, d), theta in zip(DH, thetas):
        T = T @ dh_transform(alpha, a, d, theta)
        chain.append(T.copy())
    return T, chain


def fk_com_offset(thetas_deg):
    return fk(np.array(thetas_deg, dtype=float) + JOINT_OFFSET)


# FUNÇÕES AUXILIARES

def rotation_to_rpy(R):
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        roll  = np.degrees(np.
        arctan2( R[2,1], R[2,2]))
        pitch = np.degrees(np.arctan2(-R[2,0], sy))
        yaw   = np.degrees(np.arctan2( R[1,0], R[0,0]))
    else:
        roll  = np.degrees(np.arctan2(-R[1,2], R[1,1]))
        pitch = np.degrees(np.arctan2(-R[2,0], sy))
        yaw   = 0.0
    return np.array([roll, pitch, yaw])

def euler_xyz_to_matrix(rpy_deg):
    rx, ry, rz = np.radians(rpy_deg)
    Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
    Ry = np.array([[np.cos(ry),0,np.sin(ry)],[0,1,0],[-np.sin(ry),0,np.cos(ry)]])
    Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
    return Rx @ Ry @ Rz

def print_T06(T, label='T06'):
    """Exibe T06 em formato legível: posição xyz + orientação RPY."""
    pos = T[:3, 3]
    rpy = rotation_to_rpy(T[:3, :3])
    print(f"  {label}:")
    print(f"    Posição (m) : x={pos[0]:.4f}  y={pos[1]:.4f}  z={pos[2]:.4f}")
    print(f"    RPY (°)     : roll={rpy[0]:.2f}  pitch={rpy[1]:.2f}  yaw={rpy[2]:.2f}")

def angular_error_deg(R_calc, R_gt):
    R_err = R_calc.T @ R_gt
    cos_a = np.clip((np.trace(R_err) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(cos_a))


# 4. CINEMÁTICA INVERSA (IK) 

def _solve_theta1(T06):
    P05 = (T06 @ np.array([0, 0, -D6, 1]) - np.array([0, 0, 0, 1]))[:3]
    px, py = P05[0], P05[1]
    r = np.sqrt(px**2 + py**2)

    if r < 1e-10:
        return []

    psi = np.arctan2(py, px)                  
    phi = np.arccos(np.clip(float(D4 / r), -1.0, 1.0)) 
    return [psi + phi + np.pi/2,                # shoulder left
            psi - phi + np.pi/2]                # shoulder right


def _solve_theta5(T06, t1):
    P06 = T06[:3, 3]
    P16z = P06[0]*np.sin(t1) - P06[1]*np.cos(t1)   # componente z de 1P6

    arg = (P16z - D4) / D6
    t5 = np.arccos(np.clip(arg, -1.0, 1.0))
    return [t5, -t5]


def _solve_theta6(T06, t1, t5):

    if abs(np.sin(t5)) < 1e-6:
        return 0.0  

    T01 = dh_transform(DH[0][0], DH[0][1], DH[0][2], t1)
    T16 = np.linalg.inv(T01) @ T06
    T61 = np.linalg.inv(T16)
    zx = T61[0, 2]   
    zy = T61[1, 2]   

    return np.arctan2(-zy / np.sin(t5), zx / np.sin(t5))  


def _compute_T14(T06, t1, t5, t6):
    T01 = dh_transform(DH[0][0], DH[0][1], DH[0][2], t1)   
    T45 = dh_transform(DH[4][0], DH[4][1], DH[4][2], t5)   
    T56 = dh_transform(DH[5][0], DH[5][1], DH[5][2], t6)   
    T16 = np.linalg.inv(T01) @ T06
    T14 = T16 @ np.linalg.inv(T45 @ T56)
    return T14


def _solve_theta3(T14):
    P13 = (T14 @ np.array([0, -D4, 0, 1]) - np.array([0, 0, 0, 1]))[:3]
    r   = np.linalg.norm(P13)
    arg = (r**2 - A2**2 - A3**2) / (2 * A2 * A3)  
    t3  = np.arccos(np.clip(arg, -1.0, 1.0))       
    return [t3, -t3]

def _solve_theta2(T14, t3):
    P13 = (T14 @ np.array([0, -D4, 0, 1]) - np.array([0, 0, 0, 1]))[:3]
    r   = np.linalg.norm(P13)
    delta   = np.arctan2(P13[1], -P13[0])                       
    epsilon = np.arcsin(np.clip(A3 * np.sin(t3) / r, -1, 1))     
    return -(delta - epsilon)                                    


def _solve_theta4(T14, t2, t3):
    T12 = dh_transform(DH[1][0], DH[1][1], DH[1][2], t2)  
    T23 = dh_transform(DH[2][0], DH[2][1], DH[2][2], t3)  
    T13 = T12 @ T23
    T34 = np.linalg.inv(T13) @ T14
    return np.arctan2(T34[1, 0], T34[0, 0])                 

def ik(T06_desired, print_resultados=False):
    T06 = np.array(T06_desired, dtype=float)
    solutions = []

    for idx1, t1 in enumerate(_solve_theta1(T06)):
        shoulder = 'left ' if idx1 == 0 else 'right'

        for idx5, t5 in enumerate(_solve_theta5(T06, t1)):
            wrist = 'down' if idx5 == 0 else 'up  '

            t6  = _solve_theta6(T06, t1, t5)
            T14 = _compute_T14(T06, t1, t5, t6)

            for idx3, t3 in enumerate(_solve_theta3(T14)):
                elbow = 'up  ' if idx3 == 0 else 'down'

                t2 = _solve_theta2(T14, t3)
                t4 = _solve_theta4(T14, t2, t3)

                sol = np.degrees([t1, t2, t3, t4, t5, t6])
                solutions.append(sol)

                if print_resultados:
                    print(f"  Sol {len(solutions):2d} | "
                          f"shoulder {shoulder} | elbow {elbow} | wrist {wrist} | "
                          f"θ = {np.round(sol, 1)}")

    return solutions


def ik_com_offset(T06_desired, print_resultados=False):
    solutions_model = ik(T06_desired, print_resultados=print_resultados)
    return [sol - JOINT_OFFSET for sol in solutions_model]


