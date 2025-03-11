import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Float32MultiArray
import numpy as np
import time

class MotorController(Node):
    def __init__(self):
        super().__init__('motor_controller')
        
        # Publisher for JointState
        self.publisher_ = self.create_publisher(JointState, 'joint_states', 10)
        # Publisher for Kp and Kq
        self.kp_kq_publisher_ = self.create_publisher(Float32MultiArray, 'pid_gain', 10)
        
        # Subscriber for commands
        self.subscription = self.create_subscription(
            String,
            'joint_commands',
            self.command_callback,
            10
        )
        
        self.timer = self.create_timer(0.05, self.publish_joint_states)
        self.publish_kp_kq = self.create_timer(0.1, self.publish_kp_kq)

        self.initial_joint_angles = np.array([
            [2.7, -2.7, -2.7, 2.7],  # Lower leg
            [-2.4, 2.4, 2.4, -2.4],  # Upper leg
            [0.0, 0.0, 0.0, 0.0]     # Hip
        ])

        self.command_joint_angles = np.array([
            [1.57, -1.57, -1.57, 1.57],  # Lower leg
            [-0.785, 0.785, 0.785, -0.785],  # Upper leg
            [-0.1, 0.1, -0.1, 0.1]   # Hip
        ])

        self.current_angles = self.initial_joint_angles.copy()
        self.target_angles = None
        self.start_time = None 
        self.transition_duration = 1.2
        self.command_type = None  
        self.last_command = None

        self.kp = 3
        self.kq = 0.1

        self.get_logger().info("Motor controller started, waiting for command...")

    def command_callback(self, msg):
        command = msg.data.strip()

        if self.target_angles is not None:
            self.get_logger().info(f"Ignoring '{command}' command, movement in progress.")
            return

        if command == self.last_command:
            self.get_logger().info(f"Ignoring duplicate command '{command}'.")
            return

        if command == "s":
            self.target_angles = self.command_joint_angles.copy()
            self.start_time = time.time() 
            self.command_type = "s"
            self.kp = 10
            self.kq = 0.5
            self.last_command = command
            self.get_logger().info("Received 's' command, transitioning to target angles.")
        elif command == "r":
            self.target_angles = self.initial_joint_angles.copy()
            self.start_time = time.time()
            self.command_type = "r"
            self.kp = 10
            self.kq = 0.5
            self.last_command = command
            self.get_logger().info("Received 'r' command, returning to initial angles.")
        else:
            self.get_logger().info("Unknown command. Only 's' (start) and 'r' (reset) are supported.")

    def apply_tanh_transition(self):
        if self.target_angles is None or self.start_time is None:
            return

        elapsed_time = time.time() - self.start_time
        phase = np.tanh(elapsed_time / self.transition_duration)

        if self.command_type == "s":
            self.current_angles = phase * self.target_angles + (1 - phase) * self.initial_joint_angles
        elif self.command_type == "r":
            self.current_angles = phase * self.target_angles + (1 - phase) * self.command_joint_angles
        else:
            return
    
        if np.allclose(self.current_angles, self.target_angles, atol=0.01):
            self.current_angles = self.target_angles
            self.target_angles = None
            self.command_type = None
            self.get_logger().info("Movement completed, ready for next command.")

    def publish_joint_states(self):
        self.apply_tanh_transition()

        msg = JointState()
        msg.name = [
            'frd', 'fld', 'rrd', 'rld',  # Lower legs
            'fru', 'flu', 'rru', 'rlu',  # Upper legs
            'frh', 'flh', 'rrh', 'rlh'   # Hips
        ]
        msg.position = self.current_angles.flatten().tolist()
        msg.velocity = []
        msg.effort = []
        self.publisher_.publish(msg)
    
    def publish_kp_kq(self):
        msg = Float32MultiArray()
        msg.data = [float(self.kp), float(self.kq)]
        self.kp_kq_publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    controller = MotorController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()