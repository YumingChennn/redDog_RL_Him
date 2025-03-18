import time
import math
import serial
import struct
import numpy as np
import threading
import rclpy
from rclpy.node import Node
from transformations import quaternion_from_euler
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from tf_transformations import quaternion_matrix

key = 0
flag = 0
buff = {}
angularVelocity = [0, 0, 0]
acceleration = [0, 0, 0]
magnetometer = [0, 0, 0]
angle_degree = [0, 0, 0]

# 定义IMU驱动节点类
def hex_to_short(raw_data):
    return list(struct.unpack("hhhh", bytearray(raw_data)))

def check_sum(list_data, check_data):
    return sum(list_data) & 0xff == check_data

def handle_serial_data(raw_data):
    global buff, key, angle_degree, magnetometer, acceleration, angularVelocity, pub_flag
    angle_flag = False
    buff[key] = raw_data

    key += 1

    if buff[0] != 0x55:
        key = 0
        return
    # According to the judgment of the data length bit, the corresponding length data can be obtained
    if key < 11:
        return
    else:
        data_buff = list(buff.values())  # Get dictionary ownership value
        if buff[1] == 0x51:
            if check_sum(data_buff[0:10], data_buff[10]):
                acceleration = [hex_to_short(data_buff[2:10])[i] / 32768.0 * 16 * 9.8 for i in range(0, 3)]
            else:
                print('0x51 Check failure')

        elif buff[1] == 0x52:
            if check_sum(data_buff[0:10], data_buff[10]):
                angularVelocity = [hex_to_short(data_buff[2:10])[i] / 32768.0 * 2000 * math.pi / 180 for i in
                                   range(0, 3)]

            else:
                print('0x52 Check failure')

        elif buff[1] == 0x53:
            if check_sum(data_buff[0:10], data_buff[10]):
                angle_degree = [hex_to_short(data_buff[2:10])[i] / 32768.0 * 180 for i in range(0, 3)]
                angle_flag = True
            else:
                print('0x53 Check failure')
        elif buff[1] == 0x54:
            if check_sum(data_buff[0:10], data_buff[10]):
                magnetometer = hex_to_short(data_buff[2:10])
            else:
                print('0x54 Check failure')
        else:
            buff = {}
            key = 0

        buff = {}
        key = 0
        return angle_flag

   
class IMUDriverNode(Node):
    def __init__(self, port_name):
        super().__init__('angle_publisher_node')

        # Bias 初始值
        self.bias_x = 0.22  
        self.bias_y = 0.13  
        self.bias_z = 0.1   

        # 低通濾波器參數（滑動窗口大小）
        self.N = 10  
        self.acc_buffer = []

        # 速度估算
        self.velocity = np.array([0.0, 0.0, 0.0])  # 初始速度
        self.prev_time = time.time()  # 上一次更新的時間

        # 初始化 IMU 訊息
        self.imu_msg = Imu()
        self.imu_msg.header.frame_id = 'imu_link'

        # 創建 IMU 數據發布器
        self.imu_pub = self.create_publisher(Imu, '/low_level_info/imu/data_raw', 1)
        self.vel_pub = self.create_publisher(Twist, '/low_level_info/imu/velocity', 1) 

        # 啟動 IMU 驅動線程
        self.driver_thread = threading.Thread(target=self.driver_loop, args=(port_name,))
        self.driver_thread.start()

    def driver_loop(self, port_name):
        try:
            wt_imu = serial.Serial(port="/dev/ttyUSB0", baudrate=115200, timeout=0.5)
            if wt_imu.isOpen():
                self.get_logger().info("\033[32mSerial port opened successfully...\033[0m")
            else:
                wt_imu.open()
                self.get_logger().info("\033[32mSerial port opened successfully...\033[0m")
        except Exception as e:
            print(e)
            self.get_logger().info("\033[31mSerial port opening failure\033[0m")
            exit(0)

        while True:
            try:
                buff_count = wt_imu.inWaiting()
            except Exception as e:
                print("exception:" + str(e))
                print("imu disconnect")
                exit(0)
            else:
                if buff_count > 0:
                    buff_data = wt_imu.read(buff_count)
                    for i in range(0, buff_count):
                        tag = handle_serial_data(buff_data[i])
                        if tag:
                            self.imu_data()

    def imu_data(self):
        global acceleration

        # **1. 讀取 IMU 加速度數據**
        acc_x = float(acceleration[0])
        acc_y = float(acceleration[1])
        acc_z = float(acceleration[2])

        # **2. 添加數據進入滑動窗口**
        self.acc_buffer.append([acc_x, acc_y, acc_z])
        if len(self.acc_buffer) > self.N:
            self.acc_buffer.pop(0)  # 保持窗口大小為 N

        # **3. 計算均值 Bias**
        if len(self.acc_buffer) == self.N:
            mean_acc = np.mean(self.acc_buffer, axis=0)  # 計算 N 筆數據的平均值
            self.bias_x, self.bias_y, self.bias_z = mean_acc

        # **4. 補償加速度**
        acc_x -= self.bias_x
        acc_y -= self.bias_y
        acc_z -= self.bias_z  # 重力補償

        # **5. 計算線性速度**
        current_time = time.time()
        dt = current_time - self.prev_time
        self.prev_time = current_time

        if dt > 0.001:  # 避免除以 0
            acc_vector = np.array([acc_x, acc_y, acc_z])  # 加速度向量
            self.velocity += acc_vector * dt  # v = v0 + at
        
        if np.linalg.norm(acc_vector) < 0.005:  # 加速度接近 0，代表靜止
            self.velocity = np.array([0.0, 0.0, 0.0])

        # **6. 發布線性速度**
        velocity_msg = Twist()
        velocity_msg.linear.x = self.velocity[0]
        velocity_msg.linear.y = self.velocity[1]
        velocity_msg.linear.z = self.velocity[2]

        # **7. 更新 IMU 訊息**
        self.imu_msg.header.stamp = self.get_clock().now().to_msg()
        self.imu_msg.linear_acceleration.x = acc_x
        self.imu_msg.linear_acceleration.y = acc_y
        self.imu_msg.linear_acceleration.z = acc_z

        # **8. 濾波後的角速度**
        self.imu_msg.angular_velocity.x = float(angularVelocity[0])
        self.imu_msg.angular_velocity.y = -float(angularVelocity[1])
        self.imu_msg.angular_velocity.z = -float(angularVelocity[2])

        angle_radian = [angle_degree[i] * math.pi / 180 for i in range(3)]
        qua = quaternion_from_euler(angle_radian[0], angle_radian[1], angle_radian[2])

        self.imu_msg.orientation.x = qua[0]
        self.imu_msg.orientation.y = qua[1]
        self.imu_msg.orientation.z = qua[2]
        self.imu_msg.orientation.w = qua[3]

        # **9. 發布 IMU 訊息**
        self.imu_pub.publish(self.imu_msg)
        self.vel_pub.publish(velocity_msg)

        # **10. 記錄輸出**
        self.get_logger().info(
            f"Filtered Acceleration: X={acc_x:.3f}, Y={acc_y:.3f}, Z={acc_z:.3f}"
        )
        self.get_logger().info(
            f"Estimated Velocity: X={self.velocity[0]:.3f}, Y={self.velocity[1]:.3f}, Z={self.velocity[2]:.3f}"
        )

def main():
    # 初始化ROS 2节点
    rclpy.init()
    node = IMUDriverNode('/dev/ttyUSB0')

    # 运行ROS 2节点
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    # 停止ROS 2节点
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()