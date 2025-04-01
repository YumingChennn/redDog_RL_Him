import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped
import numpy as np
import math
import matplotlib.pyplot as plt

class RosManager(Node):
    def __init__(self):
        super().__init__('imu_test')
        self.euler_sub = self.create_subscription(Vector3Stamped, '/filter/euler', self.euler_callback, 1)
        self.euler_ang = None
        self.prev_euler_ang = None
        self.publisher_ = self.create_publisher(Vector3Stamped, '/imu/euler_to_acc', 10)
        self.ang_vel = np.array([0, 0, 0])
        self.counter = 0
        self.ang_vel_list = []  # To store angular velocities

    def euler_callback(self, msg: Vector3Stamped):
        """Callback to process incoming Euler angles."""
        self.euler_ang = msg
        ang_vel = self.euler2vel(msg)

        # Store angular velocity in the list
        self.ang_vel_list.append(self.ang_vel)

        # Publish angular velocity
        ang_vel_msg = Vector3Stamped()
        ang_vel_msg.vector.x = float(ang_vel[0])
        ang_vel_msg.vector.y = float(ang_vel[1])
        ang_vel_msg.vector.z = float(ang_vel[2])
        self.publisher_.publish(ang_vel_msg)
    
    def euler2vel(self, euler: Vector3Stamped):
        """Converts Euler angles to angular velocity."""
        T = np.array([
            [1, 0, -math.sin(euler.vector.y)],
            [0, math.cos(euler.vector.x), math.cos(euler.vector.y) * math.sin(euler.vector.x)],
            [0, -math.sin(euler.vector.x), math.cos(euler.vector.x) * math.cos(euler.vector.y)]
        ])

        # Handle the first message
        if self.prev_euler_ang is None:
            self.prev_euler_ang = euler
            return self.ang_vel
        
        # Compute angular velocity from Euler angle change
        euler_ang_rate = np.array([
            euler.vector.x - self.prev_euler_ang.vector.x, 
            euler.vector.y - self.prev_euler_ang.vector.y, 
            euler.vector.z - self.prev_euler_ang.vector.z
        ])
        
        # Update angular velocity
        # self.ang_vel = T @ euler_ang_rate
        self.ang_vel = (T @ euler_ang_rate) /0.1 * (np.pi / 180)  # 轉換為 rad/s
        self.prev_euler_ang = euler

        return self.ang_vel


if __name__ == '__main__':
    rclpy.init()
    ros_manager = RosManager()
    rclpy.spin(ros_manager)
    rclpy.shutdown()