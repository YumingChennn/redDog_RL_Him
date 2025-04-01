import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading

class IMUSubscriber(Node):
    def __init__(self):
        super().__init__('imu_subscriber')
        self.subscription = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        self.angular_velocity_data = {"x": [], "y": [], "z": []}
        self.time_data = []
        self.start_time = self.get_clock().now().to_msg().sec  # 紀錄開始時間

    def imu_callback(self, msg):
        # 讀取 orientation 和 angular_velocity 數值
        angular_velocity = msg.angular_velocity

        current_time = self.get_clock().now().to_msg().sec - self.start_time  # 以秒為單位
        self.time_data.append(current_time)

        self.angular_velocity_data["x"].append(angular_velocity.x)
        self.angular_velocity_data["y"].append(angular_velocity.y)
        self.angular_velocity_data["z"].append(angular_velocity.z)
    
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
    imu_subscriber = IMUSubscriber()
    
    # 使用 threading 避免阻塞 ROS 2
    plot_thread = threading.Thread(target=plot_angular_velocity, args=(imu_subscriber,))
    plot_thread.start()

    rclpy.spin(imu_subscriber)

    imu_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
