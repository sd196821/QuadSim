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
    # print(R)
    # phi = np.arcsin(R[1, 2])
        # psi = np.arctan2(-R[1, 0] / np.cos(phi), R[1, 1] / np.cos(phi))
        # theta = np.arctan2(-R[0, 2] / np.cos(phi), R[2, 2] / np.cos(phi))
    def isRotationMatrix(Rot):
        Rt = np.transpose(Rot)
        shouldBeIdentity = np.dot(Rt, Rot)
        I = np.identity(3, dtype=Rot.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-3

    assert(isRotationMatrix(R))
    # sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    # singular = sy < 1e-6
    #
    # if not singular:
    #     phi = np.arctan2(R[2, 1], R[2, 2])
    #     theta = np.arctan2(-R[2, 0], sy)
    #     psi = np.arctan2(R[1, 0], R[0, 0])
    # else:
    #     phi = np.arctan2(-R[1, 2], R[1, 1])
    #     theta = np.arctan2(-R[2, 0], sy)
    #     psi = 0
    sy = -R[0, 2] / R[2, 2]

    singular = (np.abs(sy) >= 1145.9153)  # 1e-6
    assert (singular)
    phi = np.arcsin(R[1, 2])
    psi = np.arctan2(-R[1, 0] / np.cos(phi), R[1, 1] / np.cos(phi))
    theta = np.arctan2(-R[0, 2] / np.cos(phi), R[2, 2] / np.cos(phi))


    return np.array([phi, theta, psi])



def euler2rot(angle):
    phi, theta, psi = angle
    R = np.array([[np.cos(psi) * np.cos(theta) - np.sin(phi) * np.sin(psi) * np.sin(theta),
                   np.cos(theta) * np.sin(psi) + np.cos(psi) * np.sin(phi) * np.sin(theta),
                   -np.cos(phi) * np.sin(theta)],
                  [-np.cos(phi) * np.sin(psi), np.cos(phi) * np.cos(psi), np.sin(phi)],
                  [np.cos(psi) * np.sin(theta) + np.cos(theta) * np.sin(phi) * np.sin(psi),
                   np.sin(psi) * np.sin(theta) - np.cos(psi) * np.cos(theta) * np.sin(phi),
                   np.cos(phi) * np.cos(theta)]], dtype=np.float32)

    # R = np.array([[np.cos(psi) * np.cos(theta),
    #                np.cos(psi) * np.sin(theta) * np.sin(phi) - np.sin(psi) * np.cos(phi),
    #                np.cos(psi) * np.sin(theta) * np.cos(phi) + np.sin(psi) * np.sin(phi)],
    #               [np.sin(psi) * np.cos(theta),
    #                np.sin(psi) * np.sin(theta) * np.sin(phi) + np.cos(psi) * np.cos(phi),
    #                np.sin(psi) * np.sin(theta) * np.cos(phi) - np.cos(psi) * np.sin(phi),
    #                ],
    #               [-np.sin(theta), np.cos(theta) * np.sin(phi), np.cos(theta) * np.cos(phi)]])
    #
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
    # euler_x = np.arctan2(2*quat_w*quat_x + 2*quat_y*quat_z, quat_w*quat_w - quat_x*quat_x - quat_y*quat_y + quat_z*quat_z)
    # euler_y = -np.arcsin(2*quat_x*quat_z - 2*quat_w*quat_y)
    # euler_z = np.arctan2(2*quat_w*quat_z+2*quat_x*quat_y, quat_w*quat_w + quat_x*quat_x - quat_y*quat_y - quat_z*quat_z)
    #
    # return np.array([euler_x, euler_y, euler_z])  # in radians
    t0 = +2.0 * (quat_w * quat_x + quat_y * quat_z)
    t1 = +1.0 - 2.0 * (quat_x * quat_x + quat_y * quat_y)
    roll_x = np.arctan2(t0, t1)

    t2 = +2.0 * (quat_w * quat_y - quat_z * quat_x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = np.arcsin(t2)

    t3 = +2.0 * (quat_w * quat_z + quat_x * quat_y)
    t4 = +1.0 - 2.0 * (quat_y * quat_y + quat_z * quat_z)
    yaw_z = np.arctan2(t3, t4)

    return np.array([roll_x, pitch_y, yaw_z])



def euler2quat(euler):
    roll, pitch, yaw = euler[0], euler[1], euler[2]
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    q0 = cr * cp * cy + sr * sp * sy
    q1 = sr * cp * cy - cr * sp * sy
    q2 = cr * sp * cy + sr * cp * sy
    q3 = cr * cp * sy - sr * sp * cy
    return np.array([q0, q1, q2, q3])


def rad2deg(angle):
    return angle * 180 / np.pi


def deg2rad(angle):
    return angle * np.pi / 180

    #
    # @staticmethod
    # def quat2rot(quat):
    #     """
    #     Quaternion 2 Rotation Matrix Z-X-Y
    #     :param quat:Attitude Quaternion
    #     :return: Rotation Matrix
    #     """
    #     R = np.zeros((3, 3))
    #     quat_n = quat / np.linalg.norm(quat)
    #     qa_hat = np.zeros((3, 3))
    #     qa_hat[0, 1] = -quat_n[3]
    #     qa_hat[0, 2] = quat_n[2]
    #     qa_hat[1, 2] = -quat_n[1]
    #     qa_hat[1, 0] = quat_n[3]
    #     qa_hat[2, 0] = -quat_n[2]
    #     qa_hat[2, 1] = quat_n[1]
    #
    #     R = np.eye(3) + 2 * qa_hat * qa_hat + 2 * quat[0] * qa_hat
    #
    #     return R
    #
    # @staticmethod
    # def rot2euler(R):
    #     # print(R)
    #     def isRotationMatrix(Rot):
    #         Rt = np.transpose(Rot)
    #         shouldBeIdentity = np.dot(Rt, Rot)
    #         I = np.identity(3, dtype=Rot.dtype)
    #         n = np.linalg.norm(I - shouldBeIdentity)
    #         return n < 1e-6
    #
    #     assert(isRotationMatrix(R))
    #     # sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    #     sy = -R[0, 2]/R[2, 2]
    #
    #     singular = (np.abs(sy) >= 1145.9153) #1e-6
    #
    #     if not singular:
    #         phi = np.arcsin(R[1, 2])
    #         psi = np.arctan2(-R[1, 0] / np.cos(phi), R[1, 1] / np.cos(phi))
    #         theta = np.arctan2(-R[0, 2] / np.cos(phi), R[2, 2] / np.cos(phi))
    #     else:
    #         # phi = np.arctan2(-R[1, 2], R[1, 1])
    #         # theta = np.arctan2(-R[2, 0], sy)
    #         # psi = 0
    #         # theta = np.arctan(sy)
    #         # psi = -np.arctan2(R[1, 0], R[1, 1])
    #         # phi = 0
    #         phi = np.arcsin(R[1, 2])
    #         psi = np.arctan2(-R[1, 0] / np.cos(phi), R[1, 1] / np.cos(phi))
    #         theta = np.arctan2(-R[0, 2] / np.cos(phi), R[2, 2] / np.cos(phi))
    #
    #     # phi = np.arcsin(R[1, 2])
    #     # psi = np.arctan2(-R[1, 0] / np.cos(phi), R[1, 1] / np.cos(phi))
    #     # theta = np.arctan2(-R[0, 2] / np.cos(phi), R[2, 2] / np.cos(phi))
    #     return np.array([phi, theta, psi])
    #
    # @staticmethod
    # def euler2rot(angle):
    #     phi, theta, psi = angle
    #     R = np.array([[np.cos(psi) * np.cos(theta) - np.sin(phi) * np.sin(psi) * np.sin(theta),
    #                    np.cos(theta) * np.sin(psi) + np.cos(psi) * np.sin(phi) * np.sin(theta),
    #                    -np.cos(phi) * np.sin(theta)],
    #                   [-np.cos(phi) * np.sin(psi), np.cos(phi) * np.cos(psi), np.sin(phi)],
    #                   [np.cos(psi) * np.sin(theta) + np.cos(theta) * np.sin(phi) * np.sin(psi),
    #                    np.sin(psi) * np.sin(theta) - np.cos(psi) * np.cos(theta) * np.sin(phi),
    #                    np.cos(phi) * np.cos(theta)]], dtype=np.float32)
    #     # Z-Y-X
    #     # R = np.array([[np.cos(psi) * np.cos(theta),
    #     #                np.cos(psi) * np.sin(theta) * np.sin(phi) - np.sin(psi) * np.cos(phi),
    #     #                np.cos(psi) * np.sin(theta) * np.cos(phi) + np.sin(psi) * np.sin(phi)],
    #     #               [np.sin(psi) * np.cos(theta),
    #     #                np.sin(psi) * np.sin(theta) * np.sin(phi) + np.cos(psi) * np.cos(phi),
    #     #                np.sin(psi) * np.sin(theta) * np.cos(phi) - np.cos(psi) * np.sin(phi),
    #     #                ],
    #     #               [-np.sin(theta), np.cos(theta) * np.sin(phi), np.cos(theta) * np.cos(phi)]])
    #
    #     return R
    #
    # @staticmethod
    # def rot2quat(R):
    #     tr = R[0, 0] + R[1, 1] + R[2, 2]
    #
    #     if tr > 0:
    #         S = np.sqrt(tr + 1.0) * 2  # S = 4 * qw
    #         qw = 0.25 * S
    #         qx = (R[2, 1] - R[1, 2]) / S
    #         qy = (R[0, 2] - R[2, 0]) / S
    #         qz = (R[1, 0] - R[0, 1]) / S
    #     elif (R[0, 0] > R[1, 1]) & (R[0, 0] > R[2, 2]):
    #         S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S = 4 * qx
    #         qw = (R[2, 1] - R[1, 2]) / S
    #         qx = 0.25 * S
    #         qy = (R[0, 1] + R[1, 0]) / S
    #         qz = (R[0, 2] + R[2, 0]) / S
    #     elif R[1, 1] > R[2, 2]:
    #         S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S = 4 * qy
    #         qw = (R[0, 2] - R[2, 0]) / S
    #         qx = (R[0, 1] + R[1, 0]) / S
    #         qy = 0.25 * S
    #         qz = (R[1, 2] + R[2, 1]) / S
    #     else:
    #         S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S = 4 * qz
    #         qw = (R[1, 0] - R[0, 1]) / S
    #         qx = (R[0, 2] + R[2, 0]) / S
    #         qy = (R[1, 2] + R[2, 1]) / S
    #         qz = 0.25 * S
    #     q = np.array([qw, qx, qy, qz], dtype=np.float32)
    #     q = q * np.sign(qw)
    #     return q
    #
    # @staticmethod
    # def quat2euler(quat):
    #     """
    #     Convert Quaternion to Euler Angles
    #     """
    #     quat_w, quat_x, quat_y, quat_z = quat[0], quat[1], quat[2], quat[3]
    #     euler_x = np.arctan2(2*quat_w*quat_x + 2*quat_y*quat_z, quat_w*quat_w - quat_x*quat_x - quat_y*quat_y + quat_z*quat_z)
    #     euler_y = -np.arcsin(2*quat_x*quat_z - 2*quat_w*quat_y)
    #     euler_z = np.arctan2(2*quat_w*quat_z+2*quat_x*quat_y, quat_w*quat_w + quat_x*quat_x - quat_y*quat_y - quat_z*quat_z)
    #     return np.array([euler_x, euler_y, euler_z])
    #
    #     # t0 = +2.0 * (quat_w * quat_x + quat_y * quat_z)
    #     # t1 = +1.0 - 2.0 * (quat_x * quat_x + quat_y * quat_y)
    #     # roll_x = np.arctan2(t0, t1)
    #     #
    #     # t2 = +2.0 * (quat_w * quat_y - quat_z * quat_x)
    #     # t2 = +1.0 if t2 > +1.0 else t2
    #     # t2 = -1.0 if t2 < -1.0 else t2
    #     # pitch_y = np.arcsin(t2)
    #     #
    #     # t3 = +2.0 * (quat_w * quat_z + quat_x * quat_y)
    #     # t4 = +1.0 - 2.0 * (quat_y * quat_y + quat_z * quat_z)
    #     # yaw_z = np.arctan2(t3, t4)
    #     # return np.array([roll_x, pitch_y, yaw_z])
    #
    # @staticmethod
    # def euler2quat(euler):
    #     roll, pitch, yaw = euler[0], euler[1], euler[2]
    #     cy = np.cos(yaw * 0.5)
    #     sy = np.sin(yaw * 0.5)
    #     cp = np.cos(pitch * 0.5)
    #     sp = np.sin(pitch * 0.5)
    #     cr = np.cos(roll * 0.5)
    #     sr = np.sin(roll * 0.5)
    #
    #     q0 = cr * cp * cy + sr * sp * sy
    #     q1 = sr * cp * cy - cr * sp * sy
    #     q2 = cr * sp * cy + sr * cp * sy
    #     q3 = cr * cp * sy - sr * sp * cy
    #     return np.array([q0, q1, q2, q3])
