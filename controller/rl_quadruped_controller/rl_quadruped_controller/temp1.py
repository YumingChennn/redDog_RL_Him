import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import signal

class AngularVelocityProcessor(Node):
    def __init__(self, cutoff=1.5, fs=100, order=8):
        super().__init__('angular_velocity_processor')

        self.imu_subscription = self.create_subscription(
            Vector3Stamped,
            '/imu/angular_velocity_hr',
            self.imu_callback,
            10
        )
        self.get_logger().info("Subscribed to /imu/angular_velocity")

        self.buffer_size = max(20, order * 4)
        self.imu_buffer = {'x': [], 'y': [], 'z': []}
        self.raw_imu_log = {'x': [], 'y': [], 'z': []}  # 儲存未濾波數據
        self.filtered_imu_log = {'x': [], 'y': [], 'z': []}  # 儲存濾波後數據

        self.cutoff = cutoff
        self.fs = fs
        self.order = order
        self.b, self.a = self.butter_lowpass()

        signal.signal(signal.SIGINT, self.plot_on_exit)

    def butter_lowpass(self):
        nyquist = 0.5 * self.fs
        normal_cutoff = self.cutoff / nyquist
        b, a = butter(self.order, normal_cutoff, btype='low', analog=False)
        return b, a

    def lowpass_filter(self, data):
        if len(data) < self.buffer_size:
            return sum(data) / len(data) if len(data) > 0 else 0.0
        filtered = filtfilt(self.b, self.a, data, padlen=min(6, len(data) - 1))
        return (filtered[-1] + filtered[-2]) / 2

    def imu_callback(self, msg):
        # 記錄原始數據
        self.raw_imu_log['x'].append(msg.vector.x)
        self.raw_imu_log['y'].append(msg.vector.y)
        self.raw_imu_log['z'].append(msg.vector.z)

        self.imu_buffer['x'].append(msg.vector.x)
        self.imu_buffer['y'].append(msg.vector.y)
        self.imu_buffer['z'].append(msg.vector.z)

        for axis in ['x', 'y', 'z']:
            if len(self.imu_buffer[axis]) > self.buffer_size:
                self.imu_buffer[axis].pop(0)

        # 計算濾波後數據
        filtered_x = self.lowpass_filter(self.imu_buffer['x'])
        filtered_y = self.lowpass_filter(self.imu_buffer['y'])
        filtered_z = self.lowpass_filter(self.imu_buffer['z'])

        self.filtered_imu_log['x'].append(filtered_x)
        self.filtered_imu_log['y'].append(filtered_y)
        self.filtered_imu_log['z'].append(filtered_z)

    def plot_on_exit(self, signum, frame):
        self.get_logger().info("Ctrl+C detected, plotting IMU angular velocity data...")
        if len(self.raw_imu_log['x']) == 0:
            self.get_logger().warn("No IMU angular velocity data recorded.")
            return

        time_axis = range(len(self.raw_imu_log['x']))

        plt.figure(figsize=(12, 6))

        # 原始數據
        plt.subplot(2, 1, 1)
        plt.plot(time_axis, self.raw_imu_log['x'], label='Raw X', color='r', alpha=0.5)
        plt.plot(time_axis, self.raw_imu_log['y'], label='Raw Y', color='g', alpha=0.5)
        plt.plot(time_axis, self.raw_imu_log['z'], label='Raw Z', color='b', alpha=0.5)
        plt.xlabel('Time')
        plt.ylabel('Angular Velocity (rad/s)')
        plt.title('Raw IMU Angular Velocity')
        plt.legend()
        plt.grid()

        # 濾波後數據
        plt.subplot(2, 1, 2)
        plt.plot(time_axis, self.filtered_imu_log['x'], label='Filtered X', color='r')
        plt.plot(time_axis, self.filtered_imu_log['y'], label='Filtered Y', color='g')
        plt.plot(time_axis, self.filtered_imu_log['z'], label='Filtered Z', color='b')
        plt.xlabel('Time')
        plt.ylabel('Angular Velocity (rad/s)')
        plt.title('Filtered IMU Angular Velocity (Low-Pass)')
        plt.legend()
        plt.grid()

        plt.tight_layout()
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
