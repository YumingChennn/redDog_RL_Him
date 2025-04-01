import rclpy
from rclpy.node import Node
import matplotlib.pyplot as plt
import threading
from geometry_msgs.msg import Vector3Stamped

class AngularVSubscriber(Node):
    def __init__(self):
        super().__init__('imu_angular_velocity_subscriber')
        self.subscription = self.create_subscription(
            Vector3Stamped,
            '/imu/angular_velocity_hr',
            self.joint_state_callback,
            10
        )
        self.get_logger().info('Subscribed to angular velocity topic.')

        self.angular_velocity_data = {"x": [], "y": [], "z": []}
        self.time_data = []
        self.start_time = self.get_clock().now().to_msg().sec  # 紀錄開始時間

    def joint_state_callback(self, msg):
        current_time = self.get_clock().now().to_msg().sec - self.start_time  # 以秒為單位
        self.time_data.append(current_time)

        self.angular_velocity_data["x"].append(msg.vector.x)
        self.angular_velocity_data["y"].append(msg.vector.y)
        self.angular_velocity_data["z"].append(msg.vector.z)

    def get_Imu_data(self):
        return self.angular_velocity_data, self.time_data

def plot_angular_velocity(node):
    plt.ion()  # 開啟即時模式
    fig, ax = plt.subplots()
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)  # 非阻塞式更新 ROS 2 訂閱數據

        time_data = node.time_data
        av_data = node.angular_velocity_data

        ax.clear()
        ax.plot(time_data, av_data["x"], label="Angular Velocity X", color="r")
        ax.plot(time_data, av_data["y"], label="Angular Velocity Y", color="g")
        ax.plot(time_data, av_data["z"], label="Angular Velocity Z", color="b")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angular Velocity (rad/s)")
        ax.legend()
        ax.set_title("Real-time IMU Angular Velocity")

        plt.pause(0.1)  # 更新間隔

def main():
    rclpy.init()
    imu_subscriber = AngularVSubscriber()
    
    # 使用 threading 避免阻塞 ROS 2
    plot_thread = threading.Thread(target=plot_angular_velocity, args=(imu_subscriber,))
    plot_thread.start()

    rclpy.spin(imu_subscriber)

    imu_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
