import numpy as np

def euler_to_quaternion(roll, pitch, yaw):
    """
    將 Roll, Pitch, Yaw (度) 轉換為四元數 (qx, qy, qz, qw)
    :return: (qx, qy, qz, qw)
    """
    # 轉換為弧度
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)

    # 計算四元數
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return qx, qy, qz, qw

def quaternion_to_gravity(qx, qy, qz, qw):
    """
    將四元數 (qx, qy, qz, qw) 轉換為重力方向向量 (gx, gy, gz)
    :return: (gx, gy, gz)
    """
    gx = 2 * (qx * qz - qw * qy)
    gy = -2 * (qy * qz + qw * qx)
    gz = -1 - 2 * (qx**2 + qy**2)

    return gx, gy, gz

# # 測試輸入
# roll, pitch, yaw = 0, 0, 0  # 測試角度 (度)

# # 轉換為四元數
# qx, qy, qz, qw = euler_to_quaternion(roll, pitch, yaw)
# print(f"Quaternion: ({qx:.4f}, {qy:.4f}, {qz:.4f}, {qw:.4f})")

qx = -0.019
qy = -0.020
qz = 0.327
qw = 0.94455


# 轉換為重力向量
gx, gy, gz = quaternion_to_gravity(qx, qy, qz, qw)
print(f"Gravity Vector: ({gx:.4f}, {gy:.4f}, {gz:.4f})")
