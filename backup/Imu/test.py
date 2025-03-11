import time
import math
import serial
import struct
import numpy as np
import threading
import rclpy
from rclpy.node import Node
from transformations import quaternion_from_euler
from std_msgs.msg import String
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3
from tf_transformations import quaternion_matrix
from filterpy.kalman import ExtendedKalmanFilter


class EKFVelocityEstimator:
    def __init__(self):
        self.ekf = ExtendedKalmanFilter(dim_x=3, dim_z=3)
        self.ekf.x = np.zeros(3)  # Initial velocity [vx, vy, vz]
        self.ekf.F = np.eye(3)  # Identity matrix (acceleration directly affects velocity)
        self.ekf.Q = np.eye(3) * 0.01  # Process noise covariance
        self.ekf.R = np.eye(3) * 0.1  # Measurement noise covariance
        self.ekf.H = np.eye(3)  # Direct measurement mapping
        self.ekf.P *= 1.0

    def predict(self, acceleration, dt):
        self.ekf.x += np.array(acceleration) * dt  # v = v + a * dt
        self.ekf.P = self.ekf.F @ self.ekf.P @ self.ekf.F.T + self.ekf.Q

    def update(self, measurement):
        self.ekf.update(measurement)

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

        # 创建IMU数据发布器
        self.imu_pub = self.create_publisher(Imu, 'imu/data_raw', 1)
        self.ekf = EKFVelocityEstimator()
        self.last_time = time.time()

        # 启动IMU驱动线程
        self.driver_thread = threading.Thread(target=self.driver_loop, args=(port_name,))
        self.driver_thread.start()

    def driver_loop(self, port_name):
        # 打开串口

        try:
            wt_imu = serial.Serial(port="/dev/ttyIMU", baudrate=230400, timeout=0.5)
            if wt_imu.isOpen():
                self.get_logger().info("\033[32mSerial port opened successfully...\033[0m")
            else:
                wt_imu.open()
                self.get_logger().info("\033[32mSerial port opened successfully...\033[0m")
        except Exception as e:
            print(e)
            self.get_logger().info("\033[31mSerial port opening failure\033[0m")
            exit(0)

        # 循环读取IMU数据
        while True:
            # 读取加速度计数据

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
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = 'imu_link'

        OFFSET_ANGLE_roll = 179.0
        OFFSET_ANGLE_pitch = 0.60

        # 校正誤差
        if angle_degree[0] < 0:
            angle_degree[0] += OFFSET_ANGLE_roll
        else:
            angle_degree[0] -= OFFSET_ANGLE_roll

        if angle_degree[1] < 0:
            angle_degree[1] += OFFSET_ANGLE_pitch
        else:
            angle_degree[1] -= OFFSET_ANGLE_pitch

        angle_degree[2] = angle_degree[2]


        
        imu_msg.linear_acceleration.x = float(acceleration[0])
        imu_msg.linear_acceleration.y = float(acceleration[1])
        imu_msg.linear_acceleration.z = float(acceleration[2])

        imu_msg.angular_velocity.x = float(angularVelocity[0])
        imu_msg.angular_velocity.y = float(angularVelocity[1])
        imu_msg.angular_velocity.z = float(angularVelocity[2])

        angle_radian = [angle_degree[i] * math.pi / 180 for i in range(3)]
        qua = quaternion_from_euler(angle_radian[0], angle_radian[1], angle_radian[2])

        imu_msg.orientation.x = qua[0]
        imu_msg.orientation.y = qua[1]
        imu_msg.orientation.z = qua[2]
        imu_msg.orientation.w = qua[3]

        self.ekf.predict(acceleration, dt)
        estimated_velocity = self.ekf.ekf.x
        self.get_logger().info(f"Estimated Velocity: {estimated_velocity}")
        
        # 发布IMU消息
        self.imu_pub.publish(imu_msg)

        # self.get_logger().info(str(angularVelocity[1]))
        self.get_logger().info("Published angles: Roll={}, \nPitch={}, \nYaw={}".format(angle_degree[0], angle_degree[1], angle_degree[2]))
    

def main():
    # 初始化ROS 2节点
    rclpy.init()
    node = IMUDriverNode('/dev/ttyUSB')

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