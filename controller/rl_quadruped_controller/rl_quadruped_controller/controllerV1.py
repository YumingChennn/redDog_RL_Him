import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import String, Float32MultiArray
import numpy as np
import time
import torch
import yaml
import matplotlib.pyplot as plt
import argparse

class MotorController(Node):
    def __init__(self):
        super().__init__('motor_controller')
        
        self.publisher_ = self.create_publisher(JointState, 'joint_states', 10)
        self.kp_kq_publisher_ = self.create_publisher(Float32MultiArray, 'pid_gain', 10)
        self.timer_joint = self.create_timer(0.02, self.publish_joint_states)
        self.timer_pidgain = self.create_timer(0.02, self.publish_kp_kq)

        
        # Subscriber for commands
        self.subscription = self.create_subscription(
            String,
            'joint_commands',
            self.command_callback,
            10
        )
        self.imu_subscription = self.create_subscription(
            Imu,
            '/low_level_info/imu/data_raw',
            self.imu_callback,
            10
        )

        self.joint_state_subscription = self.create_subscription(
            JointState,
            '/low_level_info/joint_states',
            self.joint_state_callback,
            10
        )

        self.initial_joint_angles = np.array([
            [ 0.0,   0.0,  0.0,   0.0 ],
            [ 1.6, - 1.6, -1.6,   1.6 ], 
            [  -3,     3,    3,    -3 ], 
        ])

        self.default_joint_angles = np.array([
            [  0.0,   0.0,   0.0, 0.0],     # Hip
            [  2.4,  -2.4, -2.4,  2.4],  # Upper leg
            [ -2.7,   2.7,  2.7, -2.7]  # Lower leg

        ])

        self.command_joint_angles = np.array([
            [    0.1,   -0.1,    0.1,   -0.1],   # Hip
            [  0.785, -0.785, -0.785,  0.785],  # Upper leg
            [  -1.57,   1.57,   1.57,  -1.57]  # Lower leg

        ])

        self.current_angles = self.initial_joint_angles.copy()
        self.target_angles = None
        self.start_time = None 
        self.transition_duration = 1.2
        self.command_type = None  
        self.last_command = None

        self.jointAngle_data = {}
        self.jointVelocity_data = {}
        self.dof_pos = []
        self.dof_vel = []

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
    
    def imu_callback(self, msg):
        self.orientation = msg.orientation
        self.augular_velocity = msg.angular_velocity
    
    def joint_state_callback(self, msg):
        joint_names = msg.name
        positions = msg.position
        velocities = msg.velocity

        for i, name in enumerate(joint_names):
            self.jointAngle_data[name] = positions[i]
            self.jointVelocity_data[name] = velocities[i]
        
        self.dof_pos = [self.jointAngle_data['flh'],  self.jointAngle_data['flu'],  self.jointAngle_data['fld'],  
                        self.jointAngle_data['frh'],  self.jointAngle_data['fru'],  self.jointAngle_data['frd'], 
                        self.jointAngle_data['rlh'],  self.jointAngle_data['rlu'],  self.jointAngle_data['rld'],    
                        self.jointAngle_data['rrh'],  self.jointAngle_data['rru'],  self.jointAngle_data['rrd']]
    
        self.dof_vel = [self.jointVelocity_data['flh'],  self.jointVelocity_data['flu'],  self.jointVelocity_data['fld'],  
                        self.jointVelocity_data['frh'],  self.jointVelocity_data['fru'],  self.jointVelocity_data['frd'], 
                        self.jointVelocity_data['rlh'],  self.jointVelocity_data['rlu'],  self.jointVelocity_data['rld'],    
                        self.jointVelocity_data['rrh'],  self.jointVelocity_data['rru'],  self.jointVelocity_data['rrd']]

    def publish_joint_states(self):
        msg = JointState()
        msg.name = [
            'flh', 'frh', 'rlh', 'rrh',   # Hips
            'flu', 'fru', 'rlu', 'rru',  # Upper legs
            'fld', 'frd', 'rld', 'rrd'   # Lower legs
        ]
        msg.position = self.current_angles.flatten().tolist()
        msg.velocity = []
        msg.effort = []
        self.publisher_.publish(msg)
    
    def publish_kp_kq(self):
        msg = Float32MultiArray()
        msg.data = [float(self.kp), float(self.kq)]
        self.kp_kq_publisher_.publish(msg)

    def send_cmd(self):
        self.publish_joint_states()
        self.publish_kp_kq()

    def move_to_default_pos(self):
        self.kp = 3
        self.kq = 0.1
        self.current_angles = self.default_joint_angles.copy()    
        start_time = time.time()  # 記錄開始時間

        while time.time() - start_time < 15:  # 讓它跑 10 秒
            self.send_cmd()
            time.sleep(0.02)  # 50 Hz
            
        print("Moving to default pos completed.")

    def ready_to_standup(self):
        self.target_angles = self.command_joint_angles
        self.start_time = time.time()  # 確保開始時間被設置
        self.kp = 10
        self.kq = 0.4

        while True:
            elapsed_time = time.time() - self.start_time
            phase = np.tanh(elapsed_time / self.transition_duration)

            self.current_angles = phase * self.target_angles + (1 - phase) * self.default_joint_angles
            self.send_cmd()

            if np.allclose(self.current_angles, self.target_angles, atol=0.01):
                self.current_angles = self.target_angles
                self.target_angles = None
                self.command_type = None
                self.get_logger().info("Movement completed, ready for next command.")
                return  
            
            time.sleep(0.02)  # 50 Hz

            
    def run(self, config_file):
        with open(f"{config_file}", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            policy_path = config["policy_path"]
            policy = torch.jit.load(policy_path)
            xml_path = config["xml_path"]
            simulation_duration = config["simulation_duration"]
            simulation_dt = config["simulation_dt"]
            control_decimation = config["control_decimation"]

            kps = np.array(config["kps"], dtype=np.float32)
            kds = np.array(config["kds"], dtype=np.float32)

            default_angles = np.array(config["default_angles"], dtype=np.float32)

            ang_vel_scale = config["ang_vel_scale"]
            dof_pos_scale = config["dof_pos_scale"]
            dof_vel_scale = config["dof_vel_scale"]
            action_scale = config["action_scale"]
            cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

            num_actions = config["num_actions"]
            num_obs = config["num_obs"]
            one_step_obs_size = config["one_step_obs_size"]
            obs_buffer_size = config["obs_buffer_size"]
            
            cmd = np.array(config["cmd_init"], dtype=np.float32)
        
        target_dof_pos = default_angles.copy()
        action = np.zeros(num_actions, dtype=np.float32)
        obs = np.zeros(num_obs, dtype=np.float32)
    
        self.kp = 5  #10
        self.kq = 0.1 #0.4

        while True:
            self.current_angles = self.command_joint_angles.copy()
            self.send_cmd()  
            time.sleep(0.02)  # 50 Hz

        



def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file


    rclpy.init(args=args)
    controller = MotorController()

    controller.move_to_default_pos()

    controller.ready_to_standup()

    while True:
        try:
            controller.run()
        except KeyboardInterrupt:
            break
    

    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
