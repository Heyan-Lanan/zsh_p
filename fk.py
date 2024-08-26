import numpy as np


def Rx(theta):
    """Rotation matrix x-axis
            inputs a rotation angle around the x-axis and returns the rotation matrix
        Arguments:
            theta
            return Rx_matrix
    """
    Rx_matrix = np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta), -np.sin(theta), 0],
        [0, np.sin(theta), np.cos(theta), 0],
        [0, 0, 0, 1]
    ])

    return Rx_matrix


def Ry(theta):
    """Rotation matrix y-axis
            inputs a rotation angle around the y-axis and returns the rotation matrix
        Arguments:
            theta
            return Ry_matrix
    """
    Ry_matrix = np.array([
        [np.cos(theta), 0, np.sin(theta), 0],
        [0, 1, 0, 0],
        [-np.sin(theta), 0, np.cos(theta), 0],
        [0, 0, 0, 1]
    ])

    return Ry_matrix


def Rz(theta):
    """Rotation matrix for z-axis
            inputs a rotation angle around the z axis and returns the rotation matrix
        Arguments:
            theta
            return Rz_matrix
    """
    Rz_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta), np.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    return Rz_matrix


def T(dx, dy, dz):
    """Translation Matrix
            combines all the individual translation matrices into a singular function
    Arguments:
         dx, dy, dz
         return T_matrix
    """
    T_matrix = np.array([
        [1, 0, 0, dx],
        [0, 1, 0, dy],
        [0, 0, 1, dz],
        [0, 0, 0, 1]
    ])

    return T_matrix


def fk(q_values):
    q1 = q_values[0]
    q2 = q_values[1]
    q3 = q_values[2]
    q4 = q_values[3]
    q5 = q_values[4]
    q6 = q_values[5]
    q7 = q_values[6]

    E1 = T(0, 0, 0.15643)
    E2 = Rx(np.pi)
    E3 = Rz(q1)
    E4 = T(0, 0, -0.12838)
    E5 = T(0, 0.00538, 0)
    E6 = Rx(np.pi / 2)
    E7 = Rz(q2)
    E8 = T(0, 0, -0.00638)
    E9 = T(0, -0.21038, 0)
    E10 = Rx(-np.pi / 2)
    E11 = Rz(q3)
    E12 = T(0, 0, -0.21038)
    E13 = T(0, 0.00638, 0)
    E14 = Rx(np.pi / 2)
    E15 = Rz(q4)
    E16 = T(0, 0, -0.00638)
    E17 = T(0, -0.20843, 0)
    E18 = Rx(-np.pi / 2)
    E19 = Rz(q5)
    E20 = T(0, 0, -0.10593)
    E21 = Rx(np.pi / 2)
    E22 = Rz(q6)
    E23 = T(0, -0.10593, 0)
    E24 = Rx(-np.pi / 2)
    E25 = Rz(q7)
    E26 = T(0, 0, -0.06153)
    E27 = Rx(np.pi)
    E28 = T(0.0089253606274724, 0.004642492, 0.120162814)

    terms = [E1, E2, E3, E4, E5, E6, E7, E8, E9, E10,
             E11, E12, E13, E14, E15, E16, E17, E18, E19,
             E20, E21, E22, E23, E24, E25, E26, E27, E28]

    HTM = np.linalg.multi_dot(terms)

    return HTM


def fk_j6(q_values):
    q1 = q_values[0]
    q2 = q_values[1]
    q3 = q_values[2]
    q4 = q_values[3]
    q5 = q_values[4]
    q6 = q_values[5]
    q7 = q_values[6]

    E1 = T(0, 0, 0.15643)
    E2 = Rx(np.pi)
    E3 = Rz(q1)
    E4 = T(0, 0, -0.12838)
    E5 = T(0, 0.00538, 0)
    E6 = Rx(np.pi / 2)
    E7 = Rz(q2)
    E8 = T(0, 0, -0.00638)
    E9 = T(0, -0.21038, 0)
    E10 = Rx(-np.pi / 2)
    E11 = Rz(q3)
    E12 = T(0, 0, -0.21038)
    E13 = T(0, 0.00638, 0)
    E14 = Rx(np.pi / 2)
    E15 = Rz(q4)
    E16 = T(0, 0, -0.00638)
    E17 = T(0, -0.20843, 0)
    E18 = Rx(-np.pi / 2)
    E19 = Rz(q5)
    E20 = T(0, 0, -0.10593)
    E21 = Rx(np.pi / 2)
    E22 = Rz(q6)

    terms = [E1, E2, E3, E4, E5, E6, E7, E8, E9, E10,
             E11, E12, E13, E14, E15, E16, E17, E18, E19,
             E20, E21, E22]

    HTM = np.linalg.multi_dot(terms)

    return HTM


def A(dx, dy, dz):
    """Translation Matrix
            combines all the individual translation matrices into a singular function
    Arguments:
         dx, dy, dz
         return T_matrix
    """
    T_matrix = np.array([
        [0, 1, 0, dx],
        [1, 0, 0, dy],
        [0, 0, -1, dz],
        [0, 0, 0, 1]
    ])

    return T_matrix


RT_camera_tool = np.array([[-1.00000000e+00, 5.20417043e-17, -2.81719092e-15, -7.21644966e-16],
                           [-5.20417043e-17, -1.00000000e+00, -3.48332474e-15, 5.63900000e-02],
                           [-2.81719092e-15, -3.48332474e-15, 1.00000000e+00, 1.23050000e-01],
                           [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
                          )
