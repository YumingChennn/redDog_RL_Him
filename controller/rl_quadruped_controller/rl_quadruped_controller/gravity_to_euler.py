import numpy as np

def gravity_to_euler(gx, gy, gz):
    """
    將重力方向向量 (gx, gy, gz) 轉換為歐拉角 (roll, pitch, yaw)
    :return: (roll, pitch, yaw) in degrees
    """
    # 計算 Pitch (繞Y軸旋轉)
    pitch = np.arcsin(-gx)  # arcsin 輸入範圍為 [-1, 1]
    
    # 計算 Roll (繞X軸旋轉)
    roll = np.arctan2(gy, gz)
    
    # Yaw 無法從重力向量確定 (通常來自磁力計或其他來源)，這裡假設為 0
    yaw = 0.0  # 若有磁力計數據，可計算真實 yaw
    
    # 轉換為度數
    roll = np.degrees(roll)
    pitch = np.degrees(pitch)
    yaw = np.degrees(yaw)
    
    return roll, pitch, yaw

# 測試重力向量
gx, gy, gz = 0.0, 0.0, -1  # 示例值 (對應於 roll=10°, pitch=0°, yaw=0°)
roll, pitch, yaw = gravity_to_euler(gx, gy, gz)
print(f"Euler Angles: Roll={roll:.2f}°, Pitch={pitch:.2f}°, Yaw={yaw:.2f}°")