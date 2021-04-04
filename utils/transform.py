import numpy as np


def quat2rot(quat):
    """
    Quaternion 2 Rotation Matrix
    :param quat:Attitude Quaternion
    :return: Rotation Matrix
    """
    R = np.zeros((3, 3), dtype=np.float32)
    quat_n = quat / np.linalg.norm(quat)
    qa_hat = np.zeros((3, 3))
    qa_hat[0, 1] = -quat_n[3]
    qa_hat[0, 2] = quat_n[2]
    qa_hat[1, 2] = -quat_n[1]
    qa_hat[1, 0] = quat_n[3]
    qa_hat[2, 0] = -quat_n[2]
    qa_hat[2, 1] = quat_n[1]
    R = np.eye(3) + 2 * qa_hat * qa_hat + 2 * quat[0] * qa_hat
    return R


def rot2euler(R):
    def isRotationMatrix(Rot):
        Rt = np.transpose(Rot)
        shouldBeIdentity = np.dot(Rt, Rot)
        I = np.identity(3, dtype=Rot.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-3

    # assert(isRotationMatrix(R))
    if R[1, 2] < +1.0:
        phi = np.arcsin(R[1, 2])
        psi = np.arctan2(-R[1, 0], R[1, 1])
        theta = np.arctan2(-R[0, 2], R[2, 2])
        if R[1, 2] < -1.0:
            phi = -np.pi / 2.0
            psi = np.arctan2(-R[1, 0], R[1, 1])
            theta = 0
    else:
        phi = np.pi / 2.0
        psi = np.arctan2(-R[1, 0], R[1, 1])
        theta = 0


    return np.array([phi, theta, psi])



def euler2rot(angle):
    phi, theta, psi = angle[0], angle[1], angle[2]
    R = np.array([[np.cos(psi) * np.cos(theta) - np.sin(phi) * np.sin(psi) * np.sin(theta),
                   np.cos(theta) * np.sin(psi) + np.cos(psi) * np.sin(phi) * np.sin(theta),
                   -np.cos(phi) * np.sin(theta)],
                  [-np.cos(phi) * np.sin(psi), np.cos(phi) * np.cos(psi), np.sin(phi)],
                  [np.cos(psi) * np.sin(theta) + np.cos(theta) * np.sin(phi) * np.sin(psi),
                   np.sin(psi) * np.sin(theta) - np.cos(psi) * np.cos(theta) * np.sin(phi),
                   np.cos(phi) * np.cos(theta)]], dtype=np.float32)

    return R


def rot2quat(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2  # S = 4 * qw
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) & (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S = 4 * qx
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S = 4 * qy
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S = 4 * qz
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    q = np.array([qw, qx, qy, qz], dtype=np.float32)
    q = q * np.sign(qw)
    return q

def quat2euler(quat):
    """
    Convert Quaternion to Euler Angles
    """
    quat_w, quat_x, quat_y, quat_z = quat[0], quat[1], quat[2], quat[3]

    r10 = 2.0 * (quat_x * quat_y - quat_w * quat_z)
    r11 = quat_w * quat_w - quat_x * quat_x + quat_y * quat_y - quat_z * quat_z
    r12 = 2.0 * (quat_w * quat_x + quat_y * quat_z)
    r02 = 2.0 * (quat_x * quat_z - quat_w * quat_y)
    r22 = quat_w * quat_w - quat_x * quat_x - quat_y * quat_y + quat_z * quat_z

    if r12 < +1.0:
        if r12 < -1.0:
            phi = -np.pi / 2.0
            psi = np.arctan2(-r10, r11)
            theta = 0
        else:
            phi = np.arcsin(r12)
            psi = np.arctan2(-r10, r11)
            theta = np.arctan2(-r02, r22)
    else:
        phi = np.pi / 2.0
        psi = np.arctan2(-r10, r11)
        theta = 0

    return np.array([phi, theta, psi])  # in radians


def euler2quat(euler):
    roll, pitch, yaw = euler[0], euler[1], euler[2]
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    q0 = cr * cp * cy - sr * sp * sy
    q1 = sr * cp * cy - cr * sp * sy
    q2 = sr * cp * sy + cr * sp * cy
    q3 = cr * cp * sy + sr * sp * cy
    return np.array([q0, q1, q2, q3])


def rad2deg(angle):
    return angle * 180 / np.pi


def deg2rad(angle):
    return angle * np.pi / 180

