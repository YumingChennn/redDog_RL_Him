import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from scipy.signal import butter, filtfilt

class IMUSubscriber(Node):
    def __init__(self):
        super().__init__('imu_subscriber')

        # 訂閱 IMU 加速度資料
        self.subscription = self.create_subscription(
            Vector3Stamped, '/imu/acceleration_hr', self.imu_callback, 10
        )

        # 發布計算後的速度
        self.velocity_publisher = self.create_publisher(
            Vector3Stamped, '/calculated/velocity', 10
        )

        # **Bias 校正 (手動測試後固定)**
        self.bias_x, self.bias_y, self.bias_z = 0.0, 0.08, 9.79

        # **低通濾波器設定**
        self.N = 10  # 滑動窗口大小
        self.acc_buffer = deque(maxlen=self.N)

        # Butterworth 低通濾波
        self.fs = 50.0  # IMU 頻率 (假設 50Hz)
        self.cutoff = 2.0  # 截止頻率 2Hz
        self.b, self.a = butter(2, self.cutoff / (0.5 * self.fs), btype='low')

        # 速度估算
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.prev_time = None  

        # **即時數據儲存**
        self.max_len = 200  # 最多存 200 筆數據
        self.time_data = deque(maxlen=self.max_len)
        self.acc_x_data = deque(maxlen=self.max_len)
        self.acc_y_data = deque(maxlen=self.max_len)
        self.acc_z_data = deque(maxlen=self.max_len)
        self.vel_x_data = deque(maxlen=self.max_len)
        self.vel_y_data = deque(maxlen=self.max_len)
        self.vel_z_data = deque(maxlen=self.max_len)

    def butter_lowpass_filter(self, data):
        """應用 Butterworth 低通濾波"""
        return filtfilt(self.b, self.a, data)
        
    def imu_callback(self, msg):
        """處理 IMU 數據，計算速度，並發送結果"""
        acc_x, acc_y, acc_z = msg.vector.x, msg.vector.y, msg.vector.z

        # 取得 ROS 時間 (秒)
        current_time = self.get_clock().now().nanoseconds / 1e9  
        if self.prev_time is None:
            self.prev_time = current_time
            return  

        dt = current_time - self.prev_time
        self.prev_time = current_time

        if dt <= 0:
            return

        # 加入滑動窗口並應用低通濾波
        self.acc_buffer.append([acc_x, acc_y, acc_z])
        if len(self.acc_buffer) == self.N:
            acc_x, acc_y, acc_z = np.mean(self.acc_buffer, axis=0)

        # 偏差補償（動態校正）
        self.bias_x = 0.99 * self.bias_x + 0.01 * acc_x
        self.bias_y = 0.99 * self.bias_y + 0.01 * acc_y
        self.bias_z = 0.99 * self.bias_z + 0.01 * acc_z

        acc_x -= self.bias_x
        acc_y -= self.bias_y
        acc_z -= self.bias_z

        acc_vector = np.array([acc_x, acc_y, acc_z])

        # 計算增量速度 (避免直接累積)
        delta_v = acc_vector * dt

        # Zero Velocity Update (ZUPT) 增強
        if np.linalg.norm(acc_vector) < 0.05:
            self.velocity *= 0.9  # 更積極的衰減
        else:
            self.velocity += delta_v

        # 發布計算後的速度
        vel_msg = Vector3Stamped()
        vel_msg.header.stamp = self.get_clock().now().to_msg()
        vel_msg.vector.x = self.velocity[0]
        vel_msg.vector.y = self.velocity[1]
        vel_msg.vector.z = self.velocity[2]
        self.velocity_publisher.publish(vel_msg)

        # 儲存數據以供即時繪圖
        self.time_data.append(current_time)
        self.acc_x_data.append(acc_x)
        self.acc_y_data.append(acc_y)
        self.acc_z_data.append(acc_z)
        self.vel_x_data.append(self.velocity[0])
        self.vel_y_data.append(self.velocity[1])
        self.vel_z_data.append(self.velocity[2])

        self.get_logger().info(f'Acceleration: {acc_vector}, Velocity: {self.velocity}, dt={dt:.6f}s')


def plot_real_time(imu_node):
    """Matplotlib 繪製即時加速度 vs 速度 圖表"""
    fig, axes = plt.subplots(3, 1, figsize=(8, 8))

    lines = []
    for i, ax in enumerate(axes):
        ax.set_xlim(0, 5)  # X 軸初始範圍
        ax.set_ylim(-1, 1)  # 擴大 Y 軸範圍
        ax.set_ylabel(['X-axis', 'Y-axis', 'Z-axis'][i])
        ax.grid()
        line, = ax.plot([], [], label="Acceleration", color='b', linestyle='-')
        line2, = ax.plot([], [], label="Velocity", color='r', linestyle='--')
        ax.legend()
        lines.append((line, line2))

    axes[-1].set_xlabel('Time (s)')
        
    def update(frame):
        """每幀更新圖表"""
        if len(imu_node.time_data) == 0:
            return lines

        min_time = imu_node.time_data[0]
        max_time = imu_node.time_data[-1]

        for i, (line, line2) in enumerate(lines):
            acc_data = [imu_node.acc_x_data, imu_node.acc_y_data, imu_node.acc_z_data][i]
            vel_data = [imu_node.vel_x_data, imu_node.vel_y_data, imu_node.vel_z_data][i]

            time_shifted = [t - min_time for t in imu_node.time_data]
            
            # 更新加速度與速度資料
            line.set_data(time_shifted, list(acc_data))
            line2.set_data(time_shifted, list(vel_data))

            # **動態調整 X 軸範圍**
            axes[i].relim()
            axes[i].autoscale_view()

        return lines

    ani = animation.FuncAnimation(fig, update, interval=100)  
    plt.show()

def main(args=None):
    rclpy.init(args=args)
    imu_subscriber = IMUSubscriber()

    import threading
    plot_thread = threading.Thread(target=plot_real_time, args=(imu_subscriber,))
    plot_thread.start()  

    rclpy.spin(imu_subscriber)
    imu_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()