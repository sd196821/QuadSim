import numpy as np


def quat2rot(quat):
    """
    Quaternion 2 Rotation Matrix
    :param quat:Attitude Quaternion
    :return: Rotation Matrix
    """
    R = np.zeros((3, 3))
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


def quat2euler(quat):
    """
    Quaternion to Euler Angle
    :param quat:
    :return:
    """
    quat_w, quat_x, quat_y, quat_z = quat[0], quat[1], quat[2], quat[3]
    euler_x = np.arctan2(2*quat_w*quat_x + 2*quat_y*quat_z, quat_w*quat_w - quat_x*quat_x - quat_y*quat_y + quat_z*quat_z)
    euler_y = -np.arcsin(2*quat_x*quat_z - 2*quat_w*quat_y)
    euler_z = np.arctan2(2*quat_w*quat_z+2*quat_x*quat_y, quat_w*quat_w + quat_x*quat_x - quat_y*quat_y - quat_z*quat_z)
    return np.array([euler_x, euler_y, euler_z])


def rad2deg(angle):
    return angle * 180 / np.pi


def deg2rad(angle):
    return angle * np.pi / 180

