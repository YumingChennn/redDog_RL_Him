import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import signal
from math import pi

class AngularVelocityProcessor(Node):
    def __init__(self, cutoff=5.0, fs=100, order=5):
        super().__init__('angular_velocity_processor')

        # 訂閱 IMU 角速度
        self.imu_subscription = self.create_subscription(
            Vector3Stamped,
            '/imu/angular_velocity',
            self.imu_callback,
            10
        )
        # 訂閱 Euler 角
        self.euler_subscription = self.create_subscription(
            Vector3Stamped,
            '/filter/euler',
            self.euler_callback,
            10
        )
        self.get_logger().info("Subscribed to /imu/angular_velocity and /filter/euler")

        # 記錄 IMU 角速度（用於濾波）
        self.buffer_size = max(20, order * 4)
        self.imu_buffer = {'x': [], 'y': [], 'z': []}
        self.filtered_imu_velocity = None

        self.euler_angle_buffer = {'x': [], 'y': [], 'z': []}
        self.euler_angle_buffer_log = {'x': [], 'y': [], 'z': []}

        # 記錄 Euler 角度變化（用於計算角速度）
        self.prev_time = None
        self.prev_euler = None
        self.euler_angular_velocity_log = {'x': [], 'y': [], 'z': []}

        # 存儲 IMU 角速度（用於 Ctrl+C 畫圖）
        self.imu_angular_velocity_log = {'x': [], 'y': [], 'z': []}

        # Butterworth 低通濾波器
        self.cutoff = cutoff
        self.fs = fs
        self.order = order
        self.b, self.a = self.butter_lowpass()

        # 設置 Ctrl+C 退出時執行的函數
        signal.signal(signal.SIGINT, self.plot_on_exit)

    def butter_lowpass(self):
        """設置 Butterworth 低通濾波器"""
        nyquist = 0.5 * self.fs
        normal_cutoff = self.cutoff / nyquist
        b, a = butter(self.order, normal_cutoff, btype='low', analog=False)
        return b, a

    def lowpass_filter(self, data):
        """應用低通濾波器"""
        if len(data) < self.buffer_size:
            return sum(data) / len(data) if len(data) > 0 else 0.0
        filtered = filtfilt(self.b, self.a, data, padlen=min(6, len(data) - 1))
        return (filtered[-1] + filtered[-2]) / 2  # 取最後兩個點的均值

    def imu_callback(self, msg):
        """IMU 角速度數據處理與濾波"""
        # 加入緩衝區
        self.imu_buffer['x'].append(msg.vector.x)
        self.imu_buffer['y'].append(msg.vector.y)
        self.imu_buffer['z'].append(msg.vector.z)

        # 保持緩衝區大小
        for axis in ['x', 'y', 'z']:
            if len(self.imu_buffer[axis]) > self.buffer_size:
                self.imu_buffer[axis].pop(0)

        # 低通濾波
        filtered_x = self.lowpass_filter(self.imu_buffer['x'])
        filtered_y = self.lowpass_filter(self.imu_buffer['y'])
        filtered_z = self.lowpass_filter(self.imu_buffer['z'])

        # 儲存數據
        self.imu_angular_velocity_log['x'].append(filtered_x)
        self.imu_angular_velocity_log['y'].append(filtered_y)
        self.imu_angular_velocity_log['z'].append(filtered_z)

    def euler_callback(self, msg):
        """訂閱 Euler 角度並計算角速度"""
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # 轉換角度到弧度
        euler_rad = {
            'x': msg.vector.x * (pi / 180),
            'y': msg.vector.y * (pi / 180),
            'z': msg.vector.z * (pi / 180)
        }

        # 加入新數據到緩衝區
        self.euler_angle_buffer['x'].append(euler_rad['x'])
        self.euler_angle_buffer['y'].append(euler_rad['y'])
        self.euler_angle_buffer['z'].append(euler_rad['z'])
        self.euler_angle_buffer_log['x'].append(euler_rad['x'])
        self.euler_angle_buffer_log['y'].append(euler_rad['y'])
        self.euler_angle_buffer_log['z'].append(euler_rad['z'])

        # 保持緩衝區大小
        for axis in ['x', 'y', 'z']:
            if len(self.euler_angle_buffer[axis]) > self.buffer_size:
                self.euler_angle_buffer[axis].pop(0)

        # 低通濾波
        filtered_euler_x = self.lowpass_filter(self.euler_angle_buffer['x'])
        filtered_euler_y = self.lowpass_filter(self.euler_angle_buffer['y'])
        filtered_euler_z = self.lowpass_filter(self.euler_angle_buffer['z'])

        if self.prev_time is not None:
            dt = current_time - self.prev_time
            if dt > 0:
                angular_velocity = {
                    'x': (filtered_euler_x - self.prev_euler['x']) / dt,
                    'y': (filtered_euler_y - self.prev_euler['y']) / dt,
                    'z': (filtered_euler_z - self.prev_euler['z']) / dt
                }

                self.euler_angular_velocity_log['x'].append(angular_velocity['x'])
                self.euler_angular_velocity_log['y'].append(-angular_velocity['y'])
                self.euler_angular_velocity_log['z'].append(-angular_velocity['z'])

        # 更新上一個時間點的數據
        self.prev_time = current_time
        self.prev_euler = {'x': filtered_euler_x, 'y': filtered_euler_y, 'z': filtered_euler_z}

    def plot_on_exit(self, signum, frame):
        """當按下 Ctrl+C 時繪製 IMU 與 Euler 角速度變化圖"""
        self.get_logger().info("Ctrl+C detected, plotting angular velocity data...")

        # 確保有數據可用
        if len(self.imu_angular_velocity_log['x']) == 0 or len(self.euler_angular_velocity_log['x']) == 0:
            self.get_logger().warn("No data recorded.")
            return

        # 增加 Euler 角度變化圖
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))

        # IMU 角速度圖
        axs[0].plot(self.imu_angular_velocity_log['x'], label='IMU Angular Velocity X', color='r')
        axs[0].plot(self.imu_angular_velocity_log['y'], label='IMU Angular Velocity Y', color='g')
        axs[0].plot(self.imu_angular_velocity_log['z'], label='IMU Angular Velocity Z', color='b')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Angular Velocity (rad/s)')
        axs[0].set_title('IMU Angular Velocity Data (Filtered)')
        axs[0].legend()
        axs[0].grid()

        # Euler 角速度圖
        axs[1].plot(self.euler_angular_velocity_log['x'], label='Euler Angular Velocity X', color='r', linestyle='dashed')
        axs[1].plot(self.euler_angular_velocity_log['y'], label='Euler Angular Velocity Y', color='g', linestyle='dashed')
        axs[1].plot(self.euler_angular_velocity_log['z'], label='Euler Angular Velocity Z', color='b', linestyle='dashed')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Angular Velocity (rad/s)')
        axs[1].set_title('Euler Angular Velocity Data')
        axs[1].legend()
        axs[1].grid()

        # **新增的 Euler 角度變化圖**
        axs[2].plot(self.euler_angle_buffer_log['x'], label='Euler Angle X (Roll)', color='r')
        axs[2].plot(self.euler_angle_buffer_log['y'], label='Euler Angle Y (Pitch)', color='g')
        axs[2].plot(self.euler_angle_buffer_log['z'], label='Euler Angle Z (Yaw)', color='b')
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('Angle (radians)')
        axs[2].set_title('Euler Angles Over Time')
        axs[2].legend()
        axs[2].grid()

        plt.show()
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = AngularVelocityProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.plot_on_exit(None, None)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
